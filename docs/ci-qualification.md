# CI qualification and repeatable verification

Every pull request, main-branch push, manual run and Monday scheduled run
executes `.github/workflows/ci.yml` on GitHub-hosted infrastructure.
Python unit and CPU training jobs run inside pinned Python containers.
Native deployment jobs build and execute the actual Docker images on AMD64
and ARM64 hosted runners. Chromium and Qt use hosted Ubuntu runners; Qt runs
under Xvfb as a regular user so its production Chromium sandbox remains enabled.
No self-hosted runner, physical camera, desktop GPU, cloud credentials, paid
VLM requests, Hugging Face token or private capture archive is required.

## What a green check establishes

| Lane | Verifiable claim | Evidence |
| --- | --- | --- |
| Unit, Python 3.12 and 3.13 | Input validation, storage/projection semantics, provider abstraction, archive controls, material calibration, masks and depth geometry pass on both interpreters. Camera SDK protocol doubles exercise synchronization, calibration, incomplete/skewed frames, scale validation and pipeline cleanup. Worker tests verify publication, capability gating and dependency failures. | JUnit XML, statement/branch XML, JSON and HTML coverage |
| Browser | The real Flask/Waitress local API and shipped HTML/JavaScript support capture pause/resume state, interval persistence, capture once while paused, RGB/depth/IR/3D views, paired material readings, records filtering and Arrow/Parquet downloads. RGB-only cameras hide depth and volume features. Uncaught JavaScript exceptions fail the test. The existing stream suite verifies stalled/corrupt/fragmented/failed streams recover. | Synthetic Playwright traces, failure screenshots, exports and fixture logs |
| Local runtime | Both repository YOLOE artifacts initialize and infer with actual ONNX Runtime and OpenVINO CPU; their meaningful boxes/scores and prototype tensors agree within explicit numerical tolerances. Mask decoding preserves box alignment and types. A synthetic OpenVINO material encoder exercises accepted, ambiguous and rejection results, provenance and all four artifact checksum failures. The installed Orbbec SDK initializes without opening a sensor. A real botocore client validates the Bedrock request/response protocol with transport stubbed. | JUnit and runtime coverage |
| X11 desktop | The actual Qt/WebEngine console renders dashboard and records, enables depth/3D menus, adds desktop styling, and tears down while leaving an attached service healthy. | Failing exit code or success receipt in the job log |
| CPU training/export | A tiny, randomly initialized real SigLIP vision encoder learns on synthetic samples, produces finite gradients and updates both backbone and head. The production training pipeline trains and exports both frozen-head and fine-tuned variants, omits historical test rows in validation-only mode, and rejects changed features/datasets/pixels. The production exporter creates OpenVINO artifacts with maximum embedding error below `1e-4`, and the console classifier loads them and respects qualification gates. Trainer/console preprocessing matches exactly at 384 px. | JUnit and training coverage |
| Native containers, AMD64 and ARM64 | The production service and board images build and have consistent Python dependencies. The service boots under UID 1000; real concurrent Redis Lua operations deduplicate ingestion, jobs and enhancements. HTTP ingestion rejects missing/incorrect credentials, controls reject cross-origin writes, evidence remains immutable, typed exports decode, Redis outages return 503, and Redis/service restarts preserve observations, images, enhancements and settings. LiteRT EfficientDet and both YOLOE ONNX models infer offline inside the board image. | Container assertions and synthetic service log |
| Aggregate `test` | Every matrix lane succeeded. Failed, cancelled or skipped jobs cannot produce a green aggregate. | Stable check name for branch protection |

The `ci/` pytest suites reject skipped qualification tests. The normal unit suite
can retain its explicitly marked optional installed-model test; hosted runtime
tests supply synthetic material artifacts and exercise the real inference APIs
without relying on that private installation.

## Coverage policy

Coverage includes executed statements **and branches**, and includes all
`rgb`, `board` and `common` source files—even entry points with low unit coverage.
It is not an accuracy score. `scripts/check-coverage.py` enforces the aggregate
floor and separate critical-module floors in `ci/coverage-budgets.json`.
Missing report entries fail the gate; extra tests in one module cannot conceal
a regression in another gated module.

The initial aggregate floor is 75%. Camera-adapter unit coverage rose from
approximately 54% to over 75% through SDK contract tests. Storage, exports,
provider handling and several geometry/comparison modules have substantially
higher individual floors. Local-worker coverage rose from approximately 70%
to over 80%, and portable material artifact tests remove dependence on privately
installed weights. Entry points and UI/runtime integration have separate
qualification lanes; those reports do not inflate the unit coverage budget.
Raise a module's budget after adding coverage. A reduction requires an explicit
review explanation identifying the lost behavior and replacement verification.
Do not omit code or add coverage exclusions merely to clear a gate.

Actionlint checks workflow syntax, expressions and shell commands.
Syntax/undefined-name checks, strict pytest configuration/markers, dependency
consistency checks and immutable action/image pins run alongside behavioral tests.
Dependabot proposes weekly GitHub Action and browser-test dependency updates.
The scheduled full run also exposes breakage from unpinned transitive packages.

## Local reproduction

Create a test environment rather than installing both OpenCV variants into
the desktop environment:

```sh
python3 -m venv .venv-ci
.venv-ci/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv-ci/bin/python -m pytest -q --cov --cov-report=json:ci-results/coverage.json
.venv-ci/bin/python scripts/check-coverage.py ci-results/coverage.json
.venv-ci/bin/ruff check --select E9,F63,F7,F82 board rgb common scripts training ci tests
npm ci
npx playwright install --with-deps --only-shell chromium
CI_PYTHON="$PWD/.venv-ci/bin/python" npm run test:browser
CI_PYTHON="$PWD/.venv-ci/bin/python" npm run test:console
```

For actual local dependencies, use a separate environment with
`requirements-desktop.txt`, `requirements-artifacts.txt` and
`requirements-test.txt`. These install normal OpenCV through the SDK's production
dependency chain. Run `python -m pytest -q ci/test_model_runtimes.py`, then
`timeout 60s xvfb-run -a python ci/qt_console_smoke.py` as a nonroot user.
The workflow lists the Ubuntu system libraries needed by Qt and the camera SDK.

For the trainer use `requirements-training-ci.txt` in its own environment and
run `python -m pytest -q ci/test_training_runtime.py`. It installs CPU PyTorch
with the same trainer/transformer/exporter versions used by GPU training.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; the fixtures create tiny random
weights locally.

To qualify deployment containers:

```sh
docker build -t rgb-classifier:ci .
docker build -f board/Dockerfile -t rgb-board-base:ci .
docker build -f board/Dockerfile.yoloe \
  --build-arg BASE_IMAGE=rgb-board-base:ci -t rgb-board-yoloe:ci .
python3 scripts/ci-container-smoke.py
```

This script creates uniquely named containers, an internal network, an image
volume and generated ephemeral credentials. No test port is published; workloads
cannot reach the Internet. It removes its containers, volumes, network and
credential files after success or failure. Reports contain synthetic data only.
Images are built/pulled before offline execution.

## Runtime incompatibility found during qualification

The pinned OpenVINO 2026.2.1 optimizer can remove a channel Slice before
`GatherElements` in both shipped YOLOE ONNX graphs and then reject the transformed
graph's mismatched channel shapes. `board.yoloe` rewrites only the matching
static rank-three axis-one gathers into equivalent batched `GatherND` operations,
with explicit channel coordinates. Artifact files and hashes remain intact.
Runtime provenance records the replacement count. Both real models must pass
ONNX/OpenVINO numerical parity after this rewrite; a build-only check could not
have caught this failure.

## Release checks outside hosted CI

Hosted CI does not establish:

- Physical Femto Bolt USB/firmware compatibility, measured channel skew,
  reconnect reliability or sustained camera throughput.
- NUC thermal/load behavior, Intel GPU drivers, CUDA behavior or GPU memory
  limits on the training desktop.
- Detection/material accuracy or improved agreement of production weights.
  Evaluate the exact artifact on grouped held-out images using the existing
  material review/agreement process; keep teacher agreement separate from
  human-verified accuracy.
- Physical volume accuracy, calibration bias, shape assumptions, occlusion,
  reflective/transparent surfaces or absolute range accuracy. Use measured
  reference objects and record `(cyl)` estimates separately from visible proxies.
- Live Bedrock/Hugging Face permissions, connectivity, latency and quotas.
  The SDK protocol test deliberately makes no cloud request.

Record those checks with camera serial/firmware, host/GPU details, source commit,
artifact/manifest hashes, dataset identity and pass/fail tolerances in a dated
release receipt. A green CI run is the software prerequisite for that receipt.

Require the stable **`test`** check in the repository's main-branch protection.
This workflow preserves its prior name; it does not change repository protection
settings or merge a pull request.
