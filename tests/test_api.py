import httpx
import pytest

from app.main import MAX_BODY_BYTES
from app.schemas import MAX_QUESTIONS, MAX_STATE_BYTES


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
        fake_router.error = RuntimeError("CUDA out of memory: GPU 0 has paths /usr/local/lib")
        resp = client.post("/predict", json=VALID_BODY)
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Prediction failed"
        # Internal exception details must not leak to clients (#5).
        assert "CUDA" not in resp.text
        assert "/usr/local/lib" not in resp.text

    def test_503_when_router_missing(self, no_router):
        resp = no_router.post("/predict", json=VALID_BODY)
        assert resp.status_code == 503
        assert "loading" in resp.json()["detail"].lower()


class TestLimits:
    @staticmethod
    def _questions(n: int) -> dict:
        return {
            f"q{i}": {"type": "noul", "instructions": f"question {i}"}
            for i in range(n)
        }

    def test_questions_at_cap_ok(self, client):
        resp = client.post(
            "/predict", json={"state": {}, "questions": self._questions(MAX_QUESTIONS)}
        )
        assert resp.status_code == 200

    def test_questions_over_cap_is_422(self, client):
        resp = client.post(
            "/predict",
            json={"state": {}, "questions": self._questions(MAX_QUESTIONS + 1)},
        )
        assert resp.status_code == 422
        assert str(MAX_QUESTIONS) in resp.text

    def test_state_just_under_cap_ok(self, client):
        state = {"text": "x" * (MAX_STATE_BYTES - 100)}
        resp = client.post("/predict", json={"state": state, "questions": {"q": {}}})
        assert resp.status_code == 200

    def test_state_over_cap_is_422(self, client):
        state = {"text": "x" * (MAX_STATE_BYTES + 1000)}
        resp = client.post("/predict", json={"state": state, "questions": {"q": {}}})
        assert resp.status_code == 422
        assert "State too large" in resp.text

    def test_body_over_cap_is_413(self, client):
        # TestClient sets Content-Length, so the middleware rejects pre-parse.
        big = "x" * (MAX_BODY_BYTES + 1000)
        resp = client.post("/predict", json={"state": {"text": big}, "questions": {}})
        assert resp.status_code == 413
        assert "too large" in resp.text


class TestAuth:
    """LAYA_API_KEY is read per request; monkeypatch scopes it per test."""

    def test_401_without_header(self, client, monkeypatch):
        monkeypatch.setenv("LAYA_API_KEY", "secret-key")
        resp = client.post("/predict", json=VALID_BODY)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Unauthorized"

    def test_401_wrong_key(self, client, monkeypatch):
        monkeypatch.setenv("LAYA_API_KEY", "secret-key")
        resp = client.post(
            "/predict",
            headers={"Authorization": "Bearer wrong"},
            json=VALID_BODY,
        )
        assert resp.status_code == 401

    def test_200_with_correct_key(self, client, monkeypatch):
        monkeypatch.setenv("LAYA_API_KEY", "secret-key")
        resp = client.post(
            "/predict",
            headers={"Authorization": "Bearer secret-key"},
            json=VALID_BODY,
        )
        assert resp.status_code == 200

    def test_scheme_is_case_insensitive(self, client, monkeypatch):
        monkeypatch.setenv("LAYA_API_KEY", "secret-key")
        resp = client.post(
            "/predict",
            headers={"Authorization": "bearer secret-key"},
            json=VALID_BODY,
        )
        assert resp.status_code == 200

    def test_open_when_key_unset(self, client, monkeypatch):
        monkeypatch.delenv("LAYA_API_KEY", raising=False)
        resp = client.post("/predict", json=VALID_BODY)
        assert resp.status_code == 200

    def test_health_stays_open_with_key(self, client, monkeypatch):
        monkeypatch.setenv("LAYA_API_KEY", "secret-key")
        assert client.get("/health").status_code == 200


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
