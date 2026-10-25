# RGBGeneralClassifier

An Arduino Uno Q object-observation and evaluation system for work with Seattle
University students. The Portenta X8 prototype is retained for cooling experiments. The board detects COCO objects and draws live bounding
boxes. A k3s service archives observations in Redis Streams and serves a camera,
history list, and detail pane at https://rgb.wehden.com through LaminarAuth.

Each detection retains its initial classifier result and subsequent VLM readings.
The live view and archive display the latest accepted value. Confidence
thresholds, request traces, evaluation records and typed review exports are
available in the dashboard. See [review data](docs/review-data.md).

## Components

- `board/`: one V4L2 camera owner, LiteRT CPU detector, shared MJPEG cache,
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
```

For local service development use a private Redis instance with AOF persistence,
set `REDIS_URL`, `INGEST_TOKEN` (at least 32 characters), `IMAGE_DIR`, and
`SOURCES_JSON`, then run `python -m rgb.main`. The app expects ingress-level
identity enforcement. Expose local development on trusted interfaces only.

Model selection and scope are documented in [models/README.md](models/README.md).
The existing MIT copyright is preserved unchanged.
