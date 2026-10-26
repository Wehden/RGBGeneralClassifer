# RGBGeneralClassifier

An Arduino Uno Q object-observation and evaluation system for work with Seattle
University students. The Portenta X8 prototype is retained for cooling experiments. The current trial uses YOLOE-26n prompt-free ONNX with 4,585 class names. The retained first version uses EfficientDet-Lite0 COCO. Materials mode uses a separate 12-label prompted YOLOE graph. The board draws live bounding
boxes. A k3s service archives observations in Redis Streams and serves a camera,
history list, and detail pane at https://rgb.wehden.com through LaminarAuth.

Each detection retains its initial classifier result and subsequent VLM readings.
Object and Material display modes select the latest accepted value for the live
view and archive. Confidence
thresholds, request traces, evaluation records and typed review exports are
available in the dashboard. The Records page provides filters, request URLs,
and page downloads; per-camera location recording is opt in. **Pause VLM**
holds queued model work while the camera and archive keep recording.
See [review data](docs/review-data.md).

## Components

- `board/`: one V4L2 camera owner, LiteRT or YOLOE ONNX CPU detector, shared MJPEG cache,
  persistent SQLite delivery outbox, recovery after camera read failures.
- `rgb/`: Flask/Waitress web and ingest roles, Redis event archive, immutable
  content-addressed JPEGs, optional serialized Ollama second pass.
- `deploy/`: isolated k3s namespace, private Redis, NFS PVC, protected ingress,
  default-deny network policies and a camera-only ingest NodePort.
- `models/`: pinned model, extracted labels, checksum, Apache license and provenance.

See [design](docs/design.md), [operations and lab portability](docs/operations.md),
and [verification](docs/verification.md). The public repository name intentionally
retains its existing spelling, `RGBGeneralClassifer`.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q
node --test tests/js/*.test.cjs
```

For local service development use a private Redis instance with AOF persistence,
set `REDIS_URL`, `INGEST_TOKEN` (at least 32 characters), `IMAGE_DIR`, and
`SOURCES_JSON`, then run `python -m rgb.main`. The app expects ingress-level
identity enforcement. Expose local development on trusted interfaces only.

Model selection and scope are documented in [models/README.md](models/README.md).
The existing MIT copyright is preserved unchanged.

## Continuous integration

The CI workflow runs on pull requests, pushes to `main`, and manual dispatch.
It runs Python and browser behavior tests, compiles Python sources, checks both browser scripts,
verifies the pinned model checksum and builds the service image. JUnit test
results are retained as a workflow artifact for14 days. PR updates cancel an
older running check for that PR. Deployment remains manual; the workflow does
not push images or deploy to Kubernetes or the camera.

The [YOLOE trial](docs/yoloe-trial.md) records the comparison, provenance,
uncapped capture settings, rollback and 9B VLM pool integration.

See [portable networking and capture controls](docs/portable-networking.md) and
[materials mode and measured outputs](docs/materials-mode.md).
