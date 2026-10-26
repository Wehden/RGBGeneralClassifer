# RGBGeneralClassifier

The local runner uses an attached **Orbbec Femto Bolt** for color, registered
depth, and native IR, with **Nova 2 Lite on Amazon Bedrock** through the
`default` AWS profile. One Python process serves the dashboard and camera API
and keeps an SQLite archive. No Kubernetes, Redis, or Docker is needed.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/python -m rgb.local
```

Open `http://127.0.0.1:8080`. Bedrock uses `us-west-2` and
`us.amazon.nova-2-lite-v1:0`, with reasoning disabled. Camera/API reads do not
invoke Bedrock. Live detection refinement does, unless paused or started with
`--no-vlm`. The dashboard saves pause, thresholds, display mode, and explicit
Nova Lite fallback/restoration across restarts.

See [local operation and API examples](docs/local-operations.md),
[feature design](docs/local-femto-bedrock-design.md), and
[implementation verification](docs/local-verification.md).

For an X11 desktop window with the same views and review controls:

```sh
.venv/bin/pip install -r requirements-desktop.txt
.venv/bin/python -m rgb.desktop
```

Install the desktop menu entry and a command that works from any directory:

```sh
.venv/bin/python scripts/install-desktop-launcher.py
bolt-rgb
```

The console attaches to an existing service or starts a local runner. Closing
it stops only a runner it started. See [desktop setup and shortcuts](docs/desktop-console.md)
for XCB dependencies, the desktop menu launcher, native downloads and
camera-only operation.

Automatic observations default to **one per minute**, with an editable saved
interval. **Pause capture** stops automatic records and their live VLM work
while preview/detection continue; **Capture once** makes an intentional snapshot.
Camera views include RGB, Depth, IR and a streaming **3D RGB-D point cloud**
with orbit, pan, zoom and RGB/depth shading.

The local vision upgrade uses **YOLOE26 small at 640 px** and a separate
**SigLIP2 RGB-crop material pass**, with explicit uncertain results and crop
evidence. Prepare the offline models once using the
[local vision setup and benchmark guide](docs/local-vision.md). Medium remains
selectable; the original nano detector is used when the new artifacts are absent.

The retained board deployment is an Arduino Uno Q system for work with Seattle
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

The [initial image measurements](docs/bedrock-image-benchmark.md) are a short
preflight. Full crop/material-schema results and their limits are recorded in
the implementation verification document.

## Components

- `board/`: one V4L2 camera owner, LiteRT or YOLOE ONNX CPU detector, shared MJPEG cache,
  persistent SQLite delivery outbox, recovery after camera read failures.
- `rgb/`: Flask/Waitress web and ingest roles, Redis event archive, immutable
  content-addressed JPEGs, optional serialized Ollama second pass; `rgb.local`
  adds the standalone Bolt/SQLite/Bedrock path.
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
npm ci
npx playwright install --with-deps chromium
npm run test:browser
```

The live-stream browser test runs against a local synthetic camera and checks
silent stalls, closed connections, HTTP errors, corrupt JPEGs, fragmented MIME
parts, oversized responses, decode timeouts, camera switching, and cleanup. It
also exercises the default 15-second frame timeout and five-second retry delay.
To use an installed Chrome instead of Playwright's Chromium, set `CHROME_BIN`
to its executable path.

Live playback retains the last decoded frame while reconnecting, with a stale
indicator. The application buffer is capped at 4 MiB; replaced image URLs are
released. Streams with `Content-Length` display each complete JPEG immediately.
Streams without it wait for the next MIME boundary, which can add one frame of
latency. Camera status polling does not set the playback health indicator.

For local service development use a private Redis instance with AOF persistence,
set `REDIS_URL`, `INGEST_TOKEN` (at least 32 characters), `IMAGE_DIR`, and
`SOURCES_JSON`, then run `python -m rgb.main`. The app expects ingress-level
identity enforcement. Expose local development on trusted interfaces only.

Model selection and scope are documented in [models/README.md](models/README.md).
The existing MIT copyright is preserved unchanged.

## Continuous integration

The CI workflow runs on pull requests, pushes to `main`, and manual dispatch.
It runs Python tests, browser behavior tests, and Chromium live-stream fault tests,
compiles Python sources, checks browser scripts,
verifies the pinned model checksum and builds the service image. JUnit test
results are retained as a workflow artifact for14 days. PR updates cancel an
older running check for that PR. Deployment remains manual; the workflow does
not push images or deploy to Kubernetes or the camera.

The [YOLOE trial](docs/yoloe-trial.md) records the comparison, provenance,
uncapped capture settings, rollback and 9B VLM pool integration.

See [portable networking and capture controls](docs/portable-networking.md) and
[materials mode and measured outputs](docs/materials-mode.md).
