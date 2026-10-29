# Local object and material models

The local runner selects **YOLOE26 small, prompt-free, 640 px** when its
artifacts are installed. Detection targets **2 Hz**, independently of 15 Hz
sensor capture, live views, the 3D viewer, and the saved observation interval.
Medium is available through `--detector-model-dir`; the original nano graph is
the fallback when the new small graph is not installed. The API reports the
actual model, input size, artifact and vocabulary hashes, runtime, and thread
count in detector configuration and saved observations.

Both recognition stages consume **RGB**. Depth supplies box distance
summaries and the point cloud. IR remains available through views and the
camera API.

The October 28 [larger-model and volume research](local-vision-volume-research.md)
proposes paired detector/material benchmarks, use of the existing segmentation
outputs, and qualification of depth measurements against physical references.
The [first benchmark pilot](local-vision-volume-benchmarks.md) records larger-model
latency, a labeled material comparison, mask parity and native-depth SDK parity.
Physical volume accuracy remains unmeasured.

## Install once; run offline

The export environment is separate from the service. PyTorch, Transformers,
and the text encoder are needed only to prepare the artifacts:

```sh
python3 -m venv .venv-models
.venv-models/bin/pip install -r requirements-model-export.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
.venv-models/bin/python scripts/setup-local-vision.py
.venv/bin/pip install -r requirements-desktop.txt
rgb-console
```

Artifacts are placed in `~/.cache/bolt-rgb/models`. The script verifies
official small/medium checkpoint hashes and pins SigLIP2 to a specific Google
model revision. Generated manifests include hashes of every inference
artifact. Neither the service nor its classifier downloads models at startup.
Downloaded/generated weights and private camera frames are outside the repo.

For medium or the legacy detector:

```sh
rgb-console -- --detector-model-dir "$HOME/.cache/bolt-rgb/models/yoloe-26m-pf-640"
rgb-console -- --detector-model-dir models/yoloe-26n-pf-320 --detector-fps 5
rgb-console -- --no-local-materials
```

The desktop attaches to an existing service first. Model arguments apply when
it starts a service; restart the owning console to change them.
The existing `bolt-rgb` helper remains compatible with the renamed console.

## Benchmark and model selection

Measured on the attached machine: i7-1360P, approximately 32 GB RAM, four
detector threads, one private 1280×720 office frame, five warm-up calls and
30 measured calls per configuration. The existing camera service was running.
Times include input preparation and box decoding. Each model runs in a
separate process for peak-RSS measurement.

| Model | Runtime | Median | p95 | Peak RSS |
|---|---|---:|---:|---:|
| Nano, 320 px | ONNX Runtime | 54.8 ms | 74.1 ms | 340 MiB |
| Small, 640 px | ONNX Runtime | 300.2 ms | 330.4 ms | 1,558 MiB |
| Medium, 640 px | ONNX Runtime | 670.5 ms | 732.1 ms | 1,683 MiB |

Small provides the better latency tradeoff for the live console. This test
does **not** measure accuracy: the frame has no ground-truth object labels.
Medium is an explicit alternative for later paired accuracy evaluation.

OpenVINO 2026.2.1 detected only `CPU` on this installation. Both YOLOE exports
failed OpenVINO compilation at an end-to-end `GatherElements` operation.
The selected detector therefore uses ONNX Runtime; OpenVINO is used for the
SigLIP2 vision encoder, which compiled and ran successfully.

To repeat the detector comparison on private NPZ camera bundles:

```sh
.venv-models/bin/python scripts/benchmark-local-vision.py \
  --frames /private/frame-01.npz /private/frame-02.npz \
  --output /private/detector-benchmark.json
```

## SigLIP2 crop pass

Google `siglip2-base-patch16-224` runs locally through a fixed-shape FP32
OpenVINO vision encoder, with precomputed normalized text embeddings.
The initial material vocabulary is plastic, metal, glass, wood, paper,
cardboard, fabric, rubber, ceramic, leather, stone, and liquid. Human skin,
food, and an unidentifiable-image prompt are rejection candidates.

The classifier compares normalized image and text embeddings. A material is
accepted only when its cosine similarity is at least **0.20**, its lead over
the next candidate is at least **0.02**, and the winner is not a rejection
candidate. These are conservative initial policies, **not calibrated material
probabilities**. Ambiguous results remain `uncertain`; their best candidate
and scores remain inspectable. The Materials view prefers this local result,
and VLM material readings remain separately visible.

The worker holds one latest pending frame, processes at most four crops,
then waits one second. It reuses a reading only for the same continuous track,
camera session and calibration, at most five seconds old, with box IoU ≥0.7.
Crops smaller than 16 pixels on either side and stale input frames are skipped.
Undersized crops do not consume the four-crop budget or block eligible tracks.
Material work continues during VLM/capture pause without creating observations.
The crop cache is bounded to 128 assets / 16 MiB.

Saved observations retain lossless PNG evidence for each attached material
reading, its actual source frame/session/box, crop hash, model revision,
model/embedding hashes, preprocessing/prompts, match age, scores and policy.
An evicted evidence crop cannot be archived as an accepted material.
`Capture once` uses the latest fresh detector-completed frame from the current
session, calibration and image dimensions. An explicit `frame_id` retains that
identity and requires its matching detection result. Missing or stale detection
returns a retryable 503 without saving an observation; a completed result with
zero objects is still valid. Raw sensor assets refer to that observation frame.

JSON records include `local_material`; paged Parquet/Arrow downloads include
`local_material_json`. Complete schema-9 exports keep local readings in
`material_readings` with `provider=siglip2`, null probability confidence, and
separate similarity/margin/status columns. Evidence ZIPs include source crops.

## Shared UI and VLM providers

The local pipeline uses the current upstream RGB Console layout, dark forest
palette, detection/capture/VLM activity indicators, backlog display, and
capability-aware RGB/depth/IR/3D icons. The native X11 window embeds that same
dashboard. Local material status appears in the Detection segment; scores and
crop evidence remain in capture review, with uncertainty explained in Settings.

Cloud review uses `rgb.vlm_providers.ReviewProvider`, `ReviewRequest` and
`ReviewResponse` from upstream. Bedrock and OpenRouter receive the same ordered
JPEG scene/crops and available depth/IR previews through their adapters. Image
preparation, validation, serialized dispatch, pause behavior, durable job policy
and evidence are shared. SigLIP2 remains a separate offline crop worker.
The configured Bedrock Nova 2 Lite / AWS profile `default` selection is preserved;
the merge does not select OpenRouter or trigger cloud calls.

Export schema 9 combines upstream schema-8 provider auditing
(`selected_provider`, resolved model, upstream provider, cost and reasoning
tokens) with local material evidence and similarity fields. Existing SQLite
archives retain database schema 2; this export change does not rewrite captures.
Records preserve upstream camera names alongside local material readings.
Pending VLM readings use a provider-neutral label instead of a hardcoded model.
An empty local archive explains capture pause and keeps archive errors visible.

The reconciliation uses upstream main `e8a14e5`, including the merged
Redis deployment sizing and VLM backlog test-contract fixes.
Regression tests dispatch material-bearing observations through both provider
adapters without network calls, verify independent readings and exact crop
evidence, and round-trip combined provider/material fields through Arrow.

## Recorded qualification

On eight sampled object crops, 40 warm measurements gave **175.8 ms median /
203.4 ms p95** per crop with two CPU threads. The export matched the source
model: maximum preprocessing error 1.20e-7, embedding error 5.54e-7.
All eight readings were uncertain. This verifies execution and conservative
abstention; a labeled material dataset and varied camera scenes are still
needed before claiming recognition accuracy or tuning acceptance thresholds.

After activation, the normal X11 console and a second browser exercised RGB,
depth, IR and 3D against the attached camera. A 30-second check with the
material worker running measured **4.0 FPS** in the WebGL point cloud,
**7.51 ms health-request p95**, **280.8 ms detector median / 359.8 ms p95**,
and **14.70 Hz median sensor capture**. Local material readings appeared in
138 of 141 sampled detection responses. Capture remained paused, with
137 observations before and after. Existing operator settings were preserved;
upstream's two classification defaults were newly included in settings output.
No incomplete frames, skew rejections, reconnects, or browser page errors
were recorded.

Validation passed: **187 Python tests**, **20 browser unit tests**, Python
compilation, JavaScript syntax, and dependency checks. Private receipts,
model outputs and screenshots are under
`~/.local/state/bolt-rgb/local-vision-benchmark/`.

The October 27 reconciliation passed **239 Python tests** and **30 browser unit
tests**, the Chromium live-stream recovery checks, Python compilation,
JavaScript syntax checks, pinned model checksums, dependency checks and the
Docker build. The Python run included inference with the installed SigLIP2
artifacts; CI skips that optional artifact-dependent check when absent.

The reconciled dashboard was exercised against the attached Femto Bolt using a
separate private archive with both capture and VLM paused. RGB, depth, IR and
WebGL 3D switching passed without page errors; the point cloud displayed
13,054 points at 4.2 FPS with the local workers active. Intentional captures
retained material crop evidence, displayed uncalibrated scores, and appeared
in Records. The native Qt/X11 console attached to that same service, reported
the device name, and closed without stopping the attached service. No cloud
requests were dispatched. Receipts and private screenshots are under
`~/.local/state/bolt-rgb/reconciliation/`; the normal operator archive and
settings were not modified.

## Verifiable feature claims

For the learned SigLIP2 material head and partial vision fine-tuning, see the
[distillation plan](material-distillation-plan.md),
[repeatable process](material-distillation-process.md) and
[training results](material-distillation-results.md). Its acceptance policy
uses the learned head's score rather than the zero-shot cosine threshold.
When `~/.cache/bolt-rgb/models/siglip2-material-student-384/manifest.json`
exists, that deployed model becomes the local default. An explicit material
model directory overrides it; without the deployed alias, base/224 remains
the default.

| Claim | Verification |
|---|---|
| Small/medium artifacts and class mappings are verified before inference | Checksum validation and recorded manifests |
| Detector provenance reflects actual input size/runtime | Camera status and archived detector config |
| Local material work does not write observations | `test_material_worker_uses_crops_without_creating_records` |
| Undersized crops do not starve eligible tracks across passes | Parameterized undersized-crop regression |
| Ambiguous scores do not become material labels or probabilities | Rank/display/filter tests |
| Results cannot cross session/calibration/track/age/box boundaries | Parameterized stale-result tests |
| Saved readings retain exact lossless crop evidence | Asset, evidence ZIP and typed export round-trip tests |
| The GUI shows local scores independently of VLM | Browser material-reading test |
| Both provider adapters preserve local readings and exact crop evidence | `test_local_material_evidence_and_provider_audit_survive_shared_review` |
| Manual capture retains completed detection/material evidence | Manual default/pinned capture regression tests |
| Records retain camera names and material evidence together | Records and Parquet round-trip test |

Primary upstream references:

- Ultralytics YOLOE: https://docs.ultralytics.com/models/yoloe/
- Ultralytics OpenVINO export: https://docs.ultralytics.com/integrations/openvino/
- Google SigLIP2 checkpoint: https://huggingface.co/google/siglip2-base-patch16-224
- SigLIP2 paper: https://arxiv.org/abs/2502.14786
