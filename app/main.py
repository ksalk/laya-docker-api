import argparse
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette.datastructures import Headers
from starlette.types import ASGIApp

from .schemas import HealthResponse, PredictRequest

logger = logging.getLogger("laya-api")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

VALID_CHECKPOINTS = ("english", "multilingual", "typed-decisions")

router = None
runtime = {}

# Laya's Router mutates plain dict/list LRU state (load/_touch/_evict) with no
# internal locking, so concurrent predict calls are not safe. Sync handlers run
# in FastAPI's threadpool; this lock serializes inference (a single GPU would
# serialize it anyway).
_predict_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="HTTP API for the Laya decision engine")
    parser.add_argument(
        "--preload",
        default=os.environ.get("LAYA_PRELOAD", "english,multilingual"),
        help="Comma-separated checkpoints to preload: english,multilingual,typed-decisions or 'all'",
    )
    parser.add_argument(
        "--max-loaded",
        type=int,
        default=int(os.environ.get("LAYA_MAX_LOADED", "2")),
        help="Max checkpoints kept resident (LRU eviction)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("LAYA_DEVICE", "cuda"),
        help="Compute device: cuda or cpu",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LAYA_HOST", "0.0.0.0"),
        help="Bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LAYA_PORT", "8120")),
        help="Listen port",
    )
    return parser.parse_args()


def parse_preload(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(VALID_CHECKPOINTS)
    names = [n.strip().lower() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in VALID_CHECKPOINTS]
    if unknown:
        raise SystemExit(
            f"Unknown checkpoint(s): {unknown}. Valid: {list(VALID_CHECKPOINTS)} or 'all'"
        )
    if not names:
        raise SystemExit("No checkpoints given for --preload")
    return names


def vram_info():
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "used_mb": round((total - free) / 1024 / 1024),
        "total_mb": round(total / 1024 / 1024),
        "free_mb": round(free / 1024 / 1024),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, runtime

    import torch

    args = parse_args()
    preload = parse_preload(args.preload)
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        logger.info("CUDA free/total memory at startup: %.2f/%.2f GiB", free_b / 1024**3, total_b / 1024**3)
        # Rough per-checkpoint weights estimate (fp32): params * 4 bytes + overhead.
        # english/typed-decisions ~421M params, multilingual ~322M params.
        sizes = {"english": 1.8, "typed-decisions": 1.8, "multilingual": 1.4}
        needed_gb = sum(sizes.get(c, 1.5) for c in preload) + 0.6
        available_gb = free_b / 1024**3
        if needed_gb > available_gb:
            logger.warning(
                "Preloading %s needs ~%.1f GiB VRAM but only %.1f GiB is free; "
                "falling back to CPU. Reduce LAYA_PRELOAD or free VRAM to use the GPU.",
                preload, needed_gb, available_gb,
            )
            device = "cpu"

    from laya import Router

    import laya

    logger.info("Loading Laya router (preload=%s, device=%s)...", preload, device)
    start = time.perf_counter()
    router = Router(preload=False, max_loaded=args.max_loaded, device=device)
    if preload:
        router.preload(preload)
    logger.info("Router ready in %.1fs (resident: %s)", time.perf_counter() - start, router.loaded)

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    runtime.update(
        device=device,
        preload=preload,
        max_loaded=args.max_loaded,
        gpu=gpu_name,
        laya_version=laya.__version__,
        started=time.time(),
    )
    yield
    router = None


MAX_BODY_BYTES = 1024 * 1024  # 1 MB hard cap on request bodies


class _BodyTooLarge(Exception):
    """Internal signal: chunked body exceeded the size cap mid-stream."""


class BodySizeLimitMiddleware:
    """Rejects oversized request bodies with 413 before they reach the app.

    With a Content-Length header (curl, requests, httpx all send one) the
    body is never read at all. For chunked bodies it counts bytes as they
    stream and cuts off at the limit.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            if int(content_length) > self.max_bytes:
                await self._reject(send)
                return
            await self.app(scope, receive, send)
            return

        # Chunked / unknown length: count bytes as they arrive.
        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    await self._reject(send)
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, counting_receive, send)
        except _BodyTooLarge:
            pass

    async def _reject(self, send):
        body = json.dumps({"detail": f"Request body too large (max {self.max_bytes} bytes)"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


app = FastAPI(
    title="Laya Docker API",
    description="HTTP wrapper for the Laya decision engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(BodySizeLimitMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health():
    if router is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    return HealthResponse(
        status="ok",
        device=runtime["device"],
        gpu=runtime["gpu"],
        vram=vram_info(),
        checkpoints_resident=router.loaded,
        max_loaded=runtime["max_loaded"],
        laya_version=runtime["laya_version"],
    )


@app.post("/predict")
def predict(req: PredictRequest):
    if router is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    start = time.perf_counter()
    try:
        with _predict_lock:
            result = router.predict(req.state, req.questions, model=req.model)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {**result, "latency_ms": latency_ms}
