import argparse
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from .schemas import HealthResponse, PredictRequest

logger = logging.getLogger("jaya-api")

VALID_CHECKPOINTS = ("english", "multilingual", "typed-decisions")

router = None
runtime = {}


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

    args = parse_args()
    preload = parse_preload(args.preload)
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    from laya import Router

    import laya

    logger.info("Loading Laya router (preload=%s, device=%s)...", preload, device)
    start = time.perf_counter()
    router = Router(preload=preload, max_loaded=args.max_loaded, device=device)
    logger.info("Router ready in %.1fs", time.perf_counter() - start)

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


app = FastAPI(
    title="Jaya Docker API",
    description="HTTP wrapper for the Laya decision engine",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    if router is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    return HealthResponse(
        status="ok",
        device=runtime["device"],
        gpu=runtime["gpu"],
        vram=vram_info(),
        checkpoints_resident=runtime["preload"],
        max_loaded=runtime["max_loaded"],
        laya_version=runtime["laya_version"],
    )


@app.post("/predict")
async def predict(req: PredictRequest):
    if router is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    start = time.perf_counter()
    try:
        if req.model is not None:
            result = router.predict(req.state, req.questions, model=req.model)
        else:
            result = router.predict(req.state, req.questions)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {**result, "latency_ms": latency_ms}
