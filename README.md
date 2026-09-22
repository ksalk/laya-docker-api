# Laya Docker API

HTTP wrapper around the [Laya](https://github.com/NandhaKishorM/laya) decision engine, packaged as a Docker container with GPU acceleration.

Laya evaluates typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket, JSON document) in a single forward pass — tens of milliseconds on GPU, no text generation, nothing to parse or hallucinate. This repo exposes that engine over plain HTTP so you can call it from anywhere on your machine with a simple `curl` or any HTTP client.

```
HTTP client ──► :8120 (FastAPI) ──► Laya Router ──► CUDA GPU
                                       │
                                       ├─ english checkpoint
                                       ├─ multilingual checkpoint
                                       └─ typed-decisions checkpoint
```

## Prerequisites

| OS | Requirement |
|---|---|
| Linux (NVIDIA GPU) | Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Linux (no GPU) | Docker Engine only — the container falls back to CPU automatically |
| Windows | Docker Desktop with WSL2 backend + [GPU support](https://docs.docker.com/go/gpu/) |
| macOS | Docker Desktop only — NVIDIA GPUs are not passthrough-able on macOS, so it runs on CPU |

### Verify GPU passthrough (Linux / Windows)

Before building, confirm Docker can see your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this prints your GPU, you are set. On Linux, if it fails with a CDI error, install the NVIDIA Container Toolkit for your distribution (see link above), then:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

## Quickstart

```bash
git clone https://github.com/ksalk/laya-docker-api.git
cd laya-docker-api
docker compose up -d
```

The first start downloads the model weights (~1.5 GB) into `./models/`. Watch progress with:

```bash
docker compose logs -f
```

Once the log shows `Router ready`, verify:

```bash
curl http://127.0.0.1:8120/health
```

```json
{
  "status": "ok",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4060 Ti",
  "vram": {"used_mb": 4300, "total_mb": 8188, "free_mb": 3888},
  "checkpoints_resident": ["english", "multilingual"],
  "max_loaded": 2,
  "laya_version": "0.3.4"
}
```

## Configuration

Configure via environment variables (or edit `docker-compose.yml`):

| Env var | Default | Description |
|---|---|---|
| `LAYA_PRELOAD` | `english,multilingual` | Checkpoints resident at startup: comma-separated `english,multilingual,typed-decisions` or `all` |
| `LAYA_MAX_LOADED` | `2` | Max checkpoints kept in memory (LRU eviction). Non-resident checkpoints load on demand (~10 s per call) |
| `LAYA_DEVICE` | `cuda` | `cuda` or `cpu`. Falls back to CPU automatically if CUDA is unavailable |
| `LAYA_PORT` | `8120` | Listen port inside the container |

Example — CPU-only, multilingual only:

```bash
LAYA_PRELOAD=multilingual LAYA_DEVICE=cpu docker compose up -d
```

## API

### `GET /health`

Returns service status, compute device, GPU name, VRAM usage, and resident checkpoints. Returns `503` while the model is still loading.

### `POST /predict`

Request body mirrors the Laya SDK directly:

```bash
curl -X POST http://127.0.0.1:8120/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "from": "user@acme.com",
      "subject": "Duplicate charge on invoice #4411",
      "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs, outages, system errors",
          "sales": "pricing, new contracts",
          "other": "everything else"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
      },
      "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
      }
    }
  }'
```

Response — Laya's raw result plus `latency_ms`:

```json
{
  "answers": {
    "department": {"choice": "billing", "confidence": 0.94},
    "urgency": {"score": 1.84},
    "churn_risk": {"noul": 0.892}
  },
  "routing": {"model": "english", "repo": "convaiinnovations/laya/english", "reason": "..."},
  "latency_ms": 34.2
}
```

Optional fields:

- `"model": "english" | "multilingual" | "typed-decisions"` — explicit checkpoint override. When omitted, the Router auto-detects language/script per request.

### Authentication (optional)

Auth is **off by default** — the service binds to `127.0.0.1` only. To expose it beyond loopback, set `LAYA_API_KEY` and send it as a bearer token on `/predict`:

```bash
LAYA_API_KEY=my-secret docker compose up -d

curl -X POST http://127.0.0.1:8120/predict \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

Requests without a valid key get `401`. `/health` stays unauthenticated so container healthchecks work without the key.

Question types:

| Type | Output | Example use |
|---|---|---|
| `choice` | Top label + confidence | Department routing, intent classification |
| `score` | Expected ordinal level | Urgency, severity, frustration |
| `noul` | Calibrated P(true), 0.0–1.0 | Churn risk, spam, jailbreak detection |

Request limits:

| Limit | Value | Exceeding it returns |
|---|---|---|
| Questions per request | 128 | `422` with the count named in `detail` |
| Serialized `state` size | ~250 KB | `422` (`State too large: N bytes (max 250000)`) |
| Total request body | 1 MB | `413` before the body is parsed |

## VRAM tuning

Approximate weights footprint per checkpoint (fp32, plus ~0.6 GB CUDA context):

| `LAYA_PRELOAD` | ~VRAM | Notes |
|---|---|---|
| `multilingual` | ~1.9 GB | 100+ languages, weakest on English |
| `english,multilingual` | ~3.6 GB | Full language coverage — good default |
| `multilingual,typed-decisions` | ~3.6 GB | Best quality on typed workflows (tickets, invoices, triage) |
| `all` | ~5.2 GB | Everything resident; needs ~6 GB free VRAM |

If you hit CUDA OOM, reduce the preload set or switch to CPU.

## Development

Tests run locally on CPU with no GPU, no torch, and no model downloads:

```bash
mise run test
```

Requires [mise](https://mise.jdx.dev/). This creates a Python 3.12 venv in `.venv/`, installs the dev dependencies (see `requirements-dev.txt`), and runs pytest. `mise run setup` re-runs just the dependency install.

### Dependency locking

Production deps are locked in `requirements.lock` (generated from `requirements.txt`); the Dockerfile installs from the lock, so builds are reproducible. `requirements-dev.txt` stays loose and torch-free for local dev and CI. After changing `requirements.txt`, regenerate:

```bash
uv pip compile requirements.txt --universal -o requirements.lock
```

## Troubleshooting

- **`failed to discover GPU vendor from CDI`** — NVIDIA Container Toolkit not set up; see the verify step above.
- **`CUDA out of memory`** — pick a smaller `LAYA_PRELOAD` set (see VRAM tuning) or set `LAYA_DEVICE=cpu`.
- **Slow responses (~10 s) on some requests** — the request needs a non-resident checkpoint; add it to `LAYA_PRELOAD` or raise `LAYA_MAX_LOADED`.
- **`503 Model still loading`** — first start is downloading weights; follow `docker compose logs -f`.

## License

[MIT](LICENSE). The wrapped [Laya](https://github.com/NandhaKishorM/laya) decision engine itself is Apache-2.0 by [Convai Innovations](https://huggingface.co/convaiinnovations/laya).
