# Local model and depth qualification: first pilot

2026-10-28; i7-1360P NUC13, 32 GB RAM, CPU inference.
This executes the first comparisons in
[the research proposal](local-vision-volume-research.md). It is a pilot,
with separate execution, recognition-quality and physical-accuracy results.

The subsequent [blinded two-pass Nova 2 Lite comparison](box-agreement-benchmark.md)
evaluates agreement with the detector labels and bounding boxes. Use that
quality result for the initial detector comparison; the timings below
establish execution feasibility.

## Decisions supported by this run

YOLOE26 large/640 is a practical candidate for a slower recognition loop.
Its isolated median cost was only about 16% above medium in this comparison.
Neither detector's accuracy is qualified: the camera scenes have no verified
object annotations, and more detections do not demonstrate better recall.

The SigLIP2 experiments do **not** support automatically adopting a larger
encoder for materials. Base/384 improved top-choice accuracy on this sample,
but cost roughly three times as much as base/224. Large/256 performed worse
with the current prompts. Every configuration abstained under the existing
acceptance thresholds.

Calibrated native-depth backprojection agrees with the camera SDK. Analytic
volume cases pass, including negative cases for missing coverage and curved
objects. External-volume accuracy on physical objects remains unmeasured:
we need reference dimensions and an appropriate tabletop setup.

## Detector execution and masks

Each configuration ran in its own process with four ONNX Runtime inference
threads, five warm-ups and 30 measured calls. Preparation, inference and
original-image mask decoding are included. Unrelated BLAS pools used one
thread. Capture and the console were stopped for these isolated measurements.

The inputs were one previous office frame and three adjacent fresh frames,
all 1280×720. This gives two capture occasions, not four independent scenes.
The new frames contained synchronized native RGB/depth/IR with approximately
0.97 ms channel skew.

| Model, 640 px | End-to-end median | p95 | Mask decoder minimum IoU against upstream |
|---|---:|---:|---:|
| YOLOE26 small | 207.7 ms | 225.1 ms | 1.000 |
| YOLOE26 medium | 425.7 ms | 447.1 ms | 1.000 |
| YOLOE26 large | 495.2 ms | 526.0 ms | 1.000 |

Mask parity compares our NumPy/OpenCV decoder with pinned Ultralytics
8.4.188 `process_mask_native`, using the same exported detections and
prototypes. All four frames passed for all three models. This establishes
postprocessing agreement; it is not a segmentation-accuracy measurement.
An analytic test separately verifies removal of letterbox padding and
cropping into original-image coordinates.

Quantitative native-depth selection took approximately 103–105 ms median per
frame, separately from detector timing. Some detected regions had zero valid
native samples in the selected 0.5–3.86 m range. The receipts retain those
failures and describe cloud bounds as bounds of visible points, not full object
dimensions or volume.

These timings differ from the earlier live-service measurements. Compare the
models within this paired run; do not treat the historical values as a
controlled before/after speed improvement.

## Material execution and labeled pilot

The official MINC-2500 archive supplied ground-truth categories. We selected
20 images each for ceramic, fabric, glass, leather, metal, paper, plastic,
stone, water and wood. Water maps to the existing `liquid` candidate.
The deterministic selection uses the 20 lowest SHA256 values of source
image names per category. It does not filter the official train/test splits.

The encoders were used zero-shot, without training or threshold tuning on the
dataset. These public images are not guaranteed absent from model pretraining.
They evaluate a small texture/material pilot, not our camera deployment.

Each encoder also processed the same 16 private camera crops: eight box crops
and corresponding neutral-background foreground crops from small's masks.
Those crops have no material ground truth. All 16 remained uncertain for
every encoder. No recognition benefit from masking is established.

The 216 inputs ran once per encoder, after five warm-ups, using FP32 OpenVINO
CPU with two threads. Timings include preparation, inference and ranking.
A separate 30-call repeat of the same crop gave similar medians.

| SigLIP2 variant | Median per input | p95 | Top-choice accuracy, 200 MINC images | Accepted-label coverage |
|---|---:|---:|---:|---:|
| Base/224 | 152.3 ms | 159.8 ms | 35.0% | 0% |
| Base/384 | 479.5 ms | 498.7 ms | 40.5% | 0% |
| Large/256 | 680.7 ms | 699.2 ms | 24.0% | 0% |

Top choice means the highest similarity among the existing 12 material and
three rejection candidates, before acceptance gating. An `unknown` winner
counts as incorrect against a labeled material. These percentages are
**not accepted-label accuracy**: there were no accepted labels, so accepted
precision is undefined.

The unchanged policy requires similarity ≥0.20, margin ≥0.02, and a
non-rejection winner. It is uncalibrated across encoders. The larger encoder
selected `unknown` on 108/200 examples; base/224 did so on 81/200 and base/384
on 52/200. Changes in top-choice accuracy can therefore reflect rejection
behavior as well as material discrimination.

Peak process RSS was approximately 943 MiB, 960 MiB and 2,671 MiB respectively.
The larger vision encoder fits comfortably in memory on this machine.
Runtime cost and material utility, rather than parameter storage alone,
determine its suitability.

The new base/384 and large/256 exports were checked against their pinned
source models on three synthetic RGB images each. Maximum preprocessing
error was 1.20e-7; maximum normalized-embedding error was 3.30e-7 and 2.99e-7.
The service does not load the text encoder or download models during inference.

### Fixed prompt-ensemble experiment

We compared the original object-made-of wording with four fixed prompts:
the plain material name, a close-up of its surface, explicit material wording,
and the original object wording. Each prompt vector is normalized, averaged per
material, then normalized again. Rejection prompts are unchanged.
The vision graph and acceptance thresholds are identical.

The comparison used the same five images per category, 50 total, selected
from the original pilot:

| Encoder | Original prompt, matched 50 images | Ensemble, same images | Accepted coverage |
|---|---:|---:|---:|
| Base/224 | 26% | 36% | 0% |
| Large/256 | 24% | 30% | 0% |

This exploratory comparison suggests that input/prompt design deserves work.
It does not establish held-out improvement or justify relaxing thresholds.
Next, use a separate calibration set and a frozen test set, preferably with
known camera-domain materials; compare a supervised head on frozen embeddings.
The experimental artifacts are separate from the operator's existing models.

## Native depth and volume calculations

`scripts/check-native-depth.py` acquires one synchronized frame and compares
native backprojection with Orbbec's `PointCloudFilter`. It preserves the camera
singleton reservation and releases the pipeline on exit.

Across 192,206 corresponding valid native depth samples:

| Numerical comparison with SDK | Error |
|---|---:|
| Median 3D point difference | 0.0193 mm |
| p95 3D point difference | 0.0671 mm |
| Maximum 3D point difference | 0.1192 mm |
| p95 when native lens distortion is incorrectly omitted | 323.9 mm |

These are **implementation differences on the same sensor data**, not physical
sensor accuracy. Native rational Brown-Conrady distortion must be included.
The qualitative viewer's downsampled registered cloud is not the measurement
input.

Both stream-profile and pipeline SDK interfaces returned a factory
depth-to-color matrix with maximum orthogonality residual approximately
0.01127 and determinant approximately 0.99435. The quantitative helper
records this diagnostic and preserves the supplied transform for RGB mask
selection. It keeps measured points in native depth-camera coordinates,
without silently normalizing the matrix. Physical RGB/depth registration and
reference measurements remain qualification work.

The analytic tests cover:

- Millimetre-to-metre conversion, rational distortion round trip, invalid
  depths, and calibration/dimension rejection.
- A known 10 mm depth-to-color translation with points retained in native
  coordinates.
- A fully observed 200×300×100 mm cuboid: 0.006 m³, or 6,000 cm³.
- The same cuboid with 10% of footprint cells missing: an explicitly partial
  0.0054 m³ integral, no invented fill, and 90% coverage.
- An overhead sphere: the fill-to-table height-field assumption overestimates
  actual sphere volume by approximately 25%, as expected analytically.
- Robust candidate-plane recovery with noise/outliers, while leaving the
  support-surface identity unverified.

There is no physical external-volume accuracy result yet. The current office
view and unlabeled objects do not supply known volumes or a verified support
plane. Begin with a matte cuboid/cylinder at 0.6–1.0 m, independently measured
dimensions, repeated placements and visible table. Arbitrary-object volume
still requires additional views or declared shape/closure assumptions.

The subsequent [current-capture experiment](capture-range-volume-experiment.md)
reports full observed range and conditional cup/bottle cylinder envelopes,
including mask sensitivity and rejection of unsupported estimates.

## Combined live pilot

The existing `LocalRuntime` workers were exercised with camera capture,
materials and point-cloud API generation together. Each configuration used
a separate private archive with automatic capture paused and cloud review
disabled. API calls used Flask's in-process client; no socket was opened.
This checks worker/API behavior, not network latency or WebGL rendering.

| Concurrent configuration, detection target 1 Hz | Medium + base/224 | Large + base/384 |
|---|---:|---:|
| Sensor capture | 15.00 Hz | 15.00 Hz |
| Completed detection results | 1.00 Hz | 1.00 Hz |
| Detector inference median / p95 | 546.7 / 707.0 ms | 624.6 / 790.0 ms |
| Detection source-frame age p95 | 1.65 s | 1.68 s |
| Health API p95 | 1.32 ms | 1.40 ms |
| Point-cloud API p95 | 5.28 ms | 5.87 ms |
| Point-cloud requests | 4.00 Hz | 4.00 Hz |
| Responses containing any local material reading | 231/240 | 186/240 |
| Saved observations / API errors | 0 / 0 | 0 / 0 |
| Peak process RSS | 2,861 MiB | 2,862 MiB |

Material presence means any track has an attached reading, including an
uncertain reading; it does not mean every object was evaluated or any material
was correctly identified. The slower crop pass has a cost in update coverage.
Cloud review was disabled throughout.

These are 60-second smoke tests; the proposed 15-minute thermal qualification
remains outstanding. Live inference costs are greater than isolated costs,
and these data support further 1 Hz trials rather than a 2 Hz performance
promise. The live runner still uses its existing box-only detector interface;
mask decoding was exercised separately in the paired detector tests.

## Verification and remaining qualification

The full Python suite passed **247 tests**, including eight new analytic
geometry/mask tests. Python compilation, both environment dependency checks,
and whitespace checks passed. A second native SDK comparison also passed
the explicit numerical gate of p95 <1 mm and maximum <2 mm, with observed
errors again approximately 0.067 mm / 0.119 mm.

Remaining qualification is explicit: held-out object/mask annotations,
camera-domain material calibration/testing, physical volume references,
the longer sustained run, and any GPU/quantized or multi-view implementation.
The current measurements do not establish those claims.

## Reproduce and inspect

The new exporter options retain the previous default models:

```sh
.venv-models/bin/python scripts/setup-local-vision.py \
  --detectors large --material-models base-384 large-256

.venv-models/bin/python scripts/benchmark-vision-volume.py \
  --output-dir /private/qualification --stage detectors \
  --frames /private/frame-01.npz /private/frame-02.npz

.venv-models/bin/python scripts/benchmark-vision-volume.py \
  --output-dir /private/qualification \
  --prepare-minc /private/minc-2500.tar.gz

.venv-models/bin/python scripts/benchmark-vision-volume.py \
  --output-dir /private/qualification --stage materials \
  --datasets /private/qualification/minc-dataset.json \
             /private/qualification/camera-crops.json

.venv/bin/python scripts/check-native-depth.py \
  --output-dir /private/qualification/sdk-check
```

The official MINC source is linked in the research proposal. The retrieved
archive SHA256 is
`4278e074ecbee36e6768851c6d601cf71a108d4dbb552c3486ef547a59e6e5d4`.
Model revisions and release checksums are pinned in the exporter.

Private frame bundles, crop evidence, overlays, model/dataset hashes, full
readings, per-material breakdowns and live receipts are under
`~/.local/state/bolt-rgb/qualification-2026-10-28/`.
Images, downloaded datasets and generated model weights remain outside Git.
This change adds experimental qualification tools; it does not select a new
default recognition model or add a production volume endpoint.
