"""Shared fixtures: a fake Laya Router injected into app.main, plus a sync
TestClient (no network, no torch, no laya import — the app is used without
entering its lifespan, so the fake router injection is all that's needed)."""

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


class FakeRouter:
    """Stands in for laya's Router."""

    def __init__(self, result=None, error=None):
        self.result = result or {"answers": {}, "routing": {"model": "english"}}
        self.error = error
        self.calls = []
        self.route_model = None  # when set, route() decides this checkpoint

    @property
    def loaded(self):
        return ["english", "multilingual"]

    def route(self, state, questions, model=None, **kwargs):
        return {"model": self.route_model or model or "english"}

    def predict(self, state, questions, model=None, **kwargs):
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture()
def runtime_env():
    app_main.runtime.clear()
    app_main.runtime.update(
        device="cpu",
        gpu=None,
        laya_version="0.3.4-test",
        preload=["english", "multilingual"],
        started=0.0,
    )
    yield app_main.runtime
    app_main.runtime.clear()


@pytest.fixture()
def fake_router(runtime_env):
    router = FakeRouter()
    app_main.router = router
    yield router
    app_main.router = None


@pytest.fixture()
def client(fake_router):
    # Not used as a context manager on purpose: that would run the app's
    # lifespan (parse_args + laya import). The fake router is injected above.
    return TestClient(app_main.app)


@pytest.fixture()
def no_router(runtime_env):
    app_main.router = None
    return TestClient(app_main.app)
