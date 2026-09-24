"""Regression test: a slow /predict must not block the event loop.

Before the fix, `predict` was `async def` and called the blocking
`router.predict` directly on the event loop, so /health could not be served
while a predict was in flight. Now the handler is sync (threadpool) and the
event loop stays free.
"""

import asyncio
import threading
import time

import httpx
import pytest
import uvicorn

from app import main as app_main


class SlowFakeRouter:
    """Stands in for laya's Router; predict blocks like a real forward pass."""

    def __init__(self, delay: float):
        self.delay = delay

    def route(self, state, questions, model=None, **kwargs):
        return {"model": model or "english"}

    def predict(self, state, questions, model=None, **kwargs):
        time.sleep(self.delay)
        return {"answers": {}, "routing": {"model": model or "english"}}

    @property
    def loaded(self):
        return ["english"]


@pytest.fixture()
def server():
    app_main.router = SlowFakeRouter(delay=2.0)
    app_main.runtime.update(
        device="cpu",
        gpu=None,
        laya_version="0.3.4",
        preload=["english"],
    )
    # lifespan="off" skips parse_args()/laya import (finding #8); the fake
    # router injected below is all the app under test needs.
    config = uvicorn.Config(
        app_main.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    else:
        pytest.fail("uvicorn did not start")
    host, port = server.servers[0].sockets[0].getsockname()
    yield f"http://{host}:{port}"
    server.should_exit = True
    thread.join(timeout=5)
    app_main.router = None


def test_health_responds_during_slow_predict(server):
    predict_result = {}

    def run_predict():
        t0 = time.perf_counter()
        resp = httpx.post(
            f"{server}/predict",
            json={"state": {"text": "hello"}, "questions": {"q1": {"question": "hi"}}},
            timeout=10,
        )
        predict_result["resp"] = resp
        predict_result["elapsed"] = time.perf_counter() - t0

    worker = threading.Thread(target=run_predict)
    worker.start()
    time.sleep(0.5)  # let the predict enter its slow section

    t0 = time.perf_counter()
    health = httpx.get(f"{server}/health", timeout=2)
    health_elapsed = time.perf_counter() - t0

    worker.join(timeout=10)

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health_elapsed < 1.0, (
        f"/health took {health_elapsed:.2f}s during a slow predict — "
        "event loop appears blocked"
    )

    resp = predict_result["resp"]
    assert resp.status_code == 200
    assert predict_result["elapsed"] >= 2.0  # fake predict delay was honored
    assert "latency_ms" in resp.json()


def test_predict_503_when_router_missing():
    app_main.router = None
    try:

        async def call():
            transport = httpx.ASGITransport(app=app_main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await client.post(
                    "/predict",
                    json={
                        "state": {"text": "hello"},
                        "questions": {"q1": {"question": "hi"}},
                    },
                )

        resp = asyncio.run(call())
        assert resp.status_code == 503
    finally:
        app_main.router = None
