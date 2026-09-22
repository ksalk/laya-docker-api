import httpx
import pytest


VALID_BODY = {
    "state": {"subject": "Duplicate charge", "body": "We were billed twice."},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department?",
            "criteria": {"billing": "refunds"},
        }
    },
}


class TestPredictValidation:
    def test_missing_questions_is_422(self, client: httpx.Client):
        resp = client.post("/predict", json={"state": {}})
        assert resp.status_code == 422

    def test_missing_state_is_422(self, client: httpx.Client):
        resp = client.post("/predict", json={"questions": {}})
        assert resp.status_code == 422

    def test_unknown_model_is_422(self, client: httpx.Client):
        resp = client.post("/predict", json={**VALID_BODY, "model": "bogus"})
        assert resp.status_code == 422

    def test_non_dict_state_is_422(self, client: httpx.Client):
        resp = client.post("/predict", json={"state": "hello", "questions": {}})
        assert resp.status_code == 422


class TestPredictSuccess:
    def test_result_passthrough_with_latency(self, client, fake_router):
        fake_router.result = {
            "answers": {"department": {"choice": "billing"}},
            "routing": {"model": "english"},
            "usage": {"tokens": 10},
        }
        resp = client.post("/predict", json=VALID_BODY)
        assert resp.status_code == 200
        body = resp.json()
        assert body["answers"] == fake_router.result["answers"]
        assert body["routing"] == fake_router.result["routing"]
        assert body["usage"] == fake_router.result["usage"]
        assert isinstance(body["latency_ms"], (int, float))

    def test_explicit_model_forwarded_to_router(self, client, fake_router):
        client.post("/predict", json={**VALID_BODY, "model": "multilingual"})
        assert fake_router.calls[0]["model"] == "multilingual"

    def test_state_and_questions_forwarded(self, client, fake_router):
        client.post("/predict", json=VALID_BODY)
        assert fake_router.calls[0]["state"] == VALID_BODY["state"]
        assert fake_router.calls[0]["questions"] == VALID_BODY["questions"]


class TestPredictErrors:
    def test_router_exception_maps_to_500(self, client, fake_router):
        fake_router.error = RuntimeError("CUDA out of memory")
        resp = client.post("/predict", json=VALID_BODY)
        assert resp.status_code == 500
        assert "Prediction failed" in resp.json()["detail"]

    def test_503_when_router_missing(self, no_router):
        resp = no_router.post("/predict", json=VALID_BODY)
        assert resp.status_code == 503
        assert "loading" in resp.json()["detail"].lower()


class TestHealth:
    def test_ok_shape(self, client, fake_router):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["device"] == "cpu"
        assert body["gpu"] is None
        assert body["vram"] is None  # no torch installed in test venv
        assert body["checkpoints_resident"] == ["english", "multilingual"]
        assert body["max_loaded"] == 2
        assert body["laya_version"] == "0.3.4-test"

    def test_503_when_router_missing(self, no_router):
        resp = no_router.get("/health")
        assert resp.status_code == 503
        assert "loading" in resp.json()["detail"].lower()
