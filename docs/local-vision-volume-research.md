# Larger local models and depth volume: research and qualification plan

Date: 2026-10-28. Repository baseline: `99816ff` on `main`.
Status: researched proposal. The first execution, material and geometry pilot
is now recorded in [local-vision-volume-benchmarks.md](local-vision-volume-benchmarks.md).
Physical volume accuracy remains unmeasured.

Confirmed initial objective: **external object volume**, starting with tabletop
objects. Bounding dimensions support that evaluation. Container capacity is
outside the initial scope; it appears below only to distinguish measurement
semantics.

## Recommended first experiment

Compare **YOLOE26 small, medium, and large at 640 px**, and compare the current
**SigLIP2 base/224** with **base/384** and **large/256**. Decode the instance
masks already present in our detector artifacts. Use those masks for material
crop experiments and calibrated depth measurements.

Start volume qualification with opaque tabletop cuboids and cylinders of known
dimensions. Offer separate results for shape-fitted volume, bounding volume,
and, later, volume reconstructed from multiple views. Keep a single-view
height-field estimate explicitly conditional on its geometric assumptions.

The NUC has enough memory to make these experiments reasonable. Sustained
inference time, recognition quality, and sensor error decide which models and
measurements are useful. Larger weights alone do not establish better material
recognition or more accurate volume.

## Hardware and measured baseline

Read-only inspection on October 28 found:

| Component | Observed state |
|---|---|
| Processor | Intel Core i7-1360P; 12 cores, 16 logical CPUs |
| Memory | 32 GB installed; approximately 30.9 GiB reported, 26.2 GiB available during inspection |
| Graphics | Integrated Raptor Lake-P Iris Xe; no discrete GPU listed |
| Inference runtime | OpenVINO 2026.2.1 exposes only `CPU` |
| GPU prerequisites | OpenCL loader present; no OpenCL vendor ICD files; render device accessible |
| Current camera configuration | Femto Bolt; RGB 1280×720; native depth/IR 640×576; target capture 15 Hz |

Our previous measurements are recorded in [local-vision.md](local-vision.md).
They are latency measurements, not accuracy benchmarks:

| Existing configuration | Median | p95 | Peak process RSS |
|---|---:|---:|---:|
| YOLOE26 nano, 320, ONNX Runtime CPU | 54.8 ms | 74.1 ms | 340 MiB |
| YOLOE26 small, 640, ONNX Runtime CPU | 300.2 ms | 330.4 ms | 1,558 MiB |
| YOLOE26 medium, 640, ONNX Runtime CPU | 670.5 ms | 732.1 ms | 1,683 MiB |
| SigLIP2 base/224, OpenVINO CPU, per crop | 175.8 ms | 203.4 ms | Not measured separately |

Detector measurements used one office frame, four inference threads, five
warm-ups and 30 samples. The material measurement used eight crops and 40
warm samples; all eight results were uncertain under the existing acceptance
policy. Neither experiment establishes recognition accuracy.

At 670.5 ms, medium cannot sustain two serial detections per second on that
measured workload. A 1 Hz trial is sensible; larger models can run less often
or on demand. These are proposed operating rates, not measured combined
pipeline performance. Sensor capture, live rendering, recognition, and saved
observation intervals remain separate.

### Intel GPU opportunity

Intel lists Raptor Lake in its compute-runtime support table, and OpenVINO
requires the Intel graphics compute runtime/OpenCL packages for GPU inference
[1, 2]. The missing ICD is a plausible reason our installation exposes only
CPU. Diagnose the installed package versions and runtime before changing
drivers; device visibility and successful model compilation are the gates.

Test the already working SigLIP2 graph on GPU first, then compare sustained
CPU and GPU results while the 3D viewer runs. The integrated GPU shares memory,
power and rendering resources; a speedup is not established.

Both existing YOLOE exports previously failed OpenVINO CPU compilation in an
end-to-end `GatherElements` operation. Making the GPU visible does not prove
that those graphs compile. Investigate a supported raw-prediction export plus
explicit NMS, with output parity checks, before claiming an OpenVINO detector
path. Current export documentation differs from our pinned Ultralytics 8.4.188
environment: test that version's actual contract rather than copying current
flags blindly [3]. Retain the working ONNX Runtime baseline.

## Detector options

| Candidate | Role in the experiment | Principal question |
|---|---|---|
| YOLOE26 small, prompt-free, 640 | Installed control | What accuracy do we actually have? |
| YOLOE26 medium, prompt-free, 640 | Installed first comparison | Does its extra cost improve our objects and masks? |
| YOLOE26 large, prompt-free, 640 | Recommended new export | Is the quality gain worth a slower recognition loop? |
| YOLOE26 large, prompt-free, 960 | Second-stage resolution experiment | Does added image detail improve small objects and boundaries? |
| YOLOE26 extra-large, prompt-free, 640 | Optional offline comparator | Is there a meaningful gain beyond large? |
| Grounding DINO base | Optional prompted reference | Do named hard categories improve with another model family? |

Ultralytics publishes prompt-free LVIS minival detection AP of 21.4, 25.7,
27.2 and 29.9 for small, medium, large and extra-large respectively under its
end-to-end protocol at 640 px [4]. That supports testing larger variants, but
does not predict our office-scene accuracy, segmentation quality, or NUC speed.
Its non-end-to-end numbers use a different protocol; do not mix the columns.

Grounding DINO supports CPU inference and text-directed detection [5]. It
adds another inference/export path and supplies boxes rather than our existing
YOLOE masks. Use it as an offline comparison on difficult categories if the
YOLOE comparisons expose a need. Its published COCO results cannot be directly
compared with YOLOE's LVIS numbers.

A higher-resolution input also has a cost. Moving from 640 to 960 increases
image area by 2.25×; that is not a latency prediction. A larger input cannot
recover detail absent from the original frame.

### Segmentation is the shared first step

Our small and medium manifests already report two outputs:

- `[1, 100, 38]`: detection rows with boxes, scores, classes and mask coefficients.
- `[1, 32, 160, 160]`: mask prototypes.

`board/yoloe.py` currently reads only the first output and decodes the first
six values of each row. The mask branch is unused. Upstream YOLOE segmentation
checkpoints provide instance masks [4].

Decode the matching coefficients and prototypes with the upstream reference
postprocessing, then undo resize/padding into the source RGB coordinates.
Verify masks against the source model, including overlapping objects and
image edges. This gives a reusable object foreground estimate without adding
a second segmentation model initially.

Preserve the frame/session/calibration identity and the mask's model, threshold,
coordinate transform and hash. Depth alignment and mask-boundary errors need
separate inspection; a plausible mask is not a geometric accuracy guarantee.

## SigLIP2 for materials

The app already runs SigLIP2 base/224. The next experiments should test both
larger encoders and better inputs. Google supplies the fixed-resolution
variants below [6, 7].

These counts were calculated from the official configuration files by
instantiating the Transformers vision module on PyTorch's `meta` device.
No new model weights were downloaded. Counts include the pooling head; they
therefore differ from shorthand backbone sizes in the paper.

| Official model suffix | Image patch tokens | Vision parameters | FP32 parameter bytes, MiB | Priority |
|---|---:|---:|---:|---|
| `base-patch16-224` | 196 | 92,884,224 | 354.3 | Installed control |
| `base-patch16-384` | 576 | 93,176,064 | 355.4 | First |
| `large-patch16-256` | 256 | 315,956,224 | 1,205.3 | First |
| `large-patch16-384` | 576 | 316,283,904 | 1,206.5 | Second |
| `so400m-patch16-256` | 256 | 427,888,064 | 1,632.3 | Optional comparator |

All names use the prefix `google/siglip2-`. Parameter bytes are not process
RSS: activations, compiled graphs, runtime buffers and model export can add
substantial memory. The inference service needs only the vision encoder and
precomputed text vectors; loading the full image/text model is an export task.
Run candidates sequentially during qualification to avoid unnecessary
concurrent copies.

Base/384 tests increased detail with nearly unchanged parameter storage.
Large/256 tests a larger encoder at a modest input resolution. Their actual
latency must be measured: token count and transformer width/depth both matter.
Google reports improvements on general recognition/retrieval benchmarks [6],
but those are not material-recognition results for this camera.

### Crop and classification experiments

For every encoder, compare:

1. Existing rectangular crop and single material prompt.
2. Masked foreground crop with a neutral background.
3. Texture patches sampled within a slightly eroded mask.
4. Several fixed prompts per material, with a recorded aggregation rule.

Keep the rectangular crop as a control: masking removes confusing background,
but can also remove context and introduce unnatural edges. A desk chair can
contain fabric, plastic, metal and wood; evaluate the visible surface or
individual parts rather than forcing every object into one material.
Surface appearance cannot establish hidden composition or coating thickness.

Train a small linear classifier on frozen embeddings if zero-shot matching
remains weak. This adds a supervised material head without training a new
vision backbone. Split by physical object and capture session, not adjacent
frames, so a test crop is not a near duplicate of its training crop.
MINC/MINC-2500 provides a material-focused benchmark with 23 categories [8];
our own labeled Femto crops are necessary to evaluate this deployment's
lighting, vocabulary and mixed objects.

Evaluate accepted-label precision, per-material recall, coverage/abstention,
confusion matrices and mixed-material failure cases. Tune score and margin
thresholds on a calibration split, then freeze them for the test split.
Cosine similarity remains a similarity, not a probability. The existing
0.20/0.02 thresholds should not be assumed to transfer to another encoder.

The current installer hardcodes base/224, although the runtime reads a
manifest. Supporting another variant requires its pinned revision, actual
processor/input shape, text embeddings and verified exported vision graph.
Check preprocessing and embedding parity before benchmarking.

## What depth can establish about volume

The existing app provides median/percentile depth within detection boxes and
a calibrated preview point cloud. It does not measure object dimensions or
volume. Box depth includes background; use foreground masks and a support
plane for the proposed measurements.

### Sensor limits

Orbbec identifies native 640×576 as NFOV unbinned, with a 0.5–3.86 m operating
range and 75°×65° field of view. Its published typical systematic depth error
is less than 11 mm + 0.1% of distance, and random-error standard deviation is
at most 17 mm under its stated conditions [9]. These are manufacturer
specifications, not measured error on our objects.

Do not interpret raw depth units as accuracy. Averaging stable frames can
reduce random noise; it does not remove calibration bias, multipath or
segmentation errors. Transparent, reflective and poorly returning surfaces
need explicit failure cases and coverage checks.

A 1280×720 registered depth image still originates from the native 640×576
measurements. Registration does not create additional independent depth
samples. The current viewer deliberately samples with stride 6 and caps its
cloud at 30,000 points; measurement must use a separate quantitative path.

Back-project native depth with its own intrinsics/distortion and scale, then
use the depth-to-color extrinsic transform to select points with the RGB mask.
Handle occlusion at color/depth boundaries and verify the result against SDK
coordinate transforms [10]. This avoids treating replicated registered
pixels as independent measurements.

### Distinct measurement products

| Result | Method | What the result means |
|---|---|---|
| Bounding dimensions/volume | Object cloud, support plane, oriented bounds | Bounds of observed points; full-object envelope only with sufficient coverage or declared shape assumptions |
| Shape-fitted volume | Fit cuboid/cylinder/etc., check residuals | Volume conditional on the selected geometric model |
| Height-field volume | Integrate surface height on a metric support-plane grid | Volume assuming the object fills the space between observed surface and table |
| Reconstructed external volume | Multiple registered views, TSDF fusion, validated closed mesh | Volume enclosed by the reconstructed surface, including any declared closure assumptions |
| Container capacity | Observe/reconstruct the interior separately | Internal capacity; it cannot be inferred from an opaque object's exterior |

For a height field, use `V ≈ sum(height[i] × cell_area)` on a grid in table
coordinates. Image pixels do not have a constant physical footprint across
depth and perspective. Missing cells remain missing unless a documented
policy fills them; record the filled area and sensitivity to that policy.
An overhead view improves footprint visibility. Handles, overhangs, spheres
and hollow objects violate the fill-to-table assumption.

A single depth image observes a visible surface, not an entire closed object.
Unseen backs, undersides and cavities prevent a unique general volume result.
A convex hull fills concavities, and mesh completion can invent unobserved
surfaces. Neither should silently become a measured solid volume.

### Multiple views

Open3D provides CPU TSDF integration using calibrated RGB-D and per-view poses;
its mesh-volume implementation requires a watertight, orientable mesh [11, 12].
That is a practical local path to investigate after the simpler measurements.

For a static object, move the camera and estimate poses using markers or
RGB-D registration. For a fixed camera and turntable, transform each view into
the object's reference frame using the calibrated turntable axis and angle.
Camera odometry on the stationary background alone does not recover the
rotating object's pose. The camera IMU alone does not supply a complete
translation trajectory.

Use a bounded object region, inspect pose residuals, fuse views, and measure
sensitivity to voxel size and view count. Allocate an explicit memory budget
and reject incomplete scans. TSDF reconstruction is feasible to prototype on
CPU; interactive throughput is unmeasured. A closed mesh is a necessary
geometric check, not proof that its completed surfaces match the object.
The underside still requires another view or an explicit support-plane cap.

### Accuracy experiment

Start at 0.6–1.0 m in the current NFOV mode with matte opaque references
roughly 15–40 cm across, with the table visible. Include cuboids and cylinders
with independently measured dimensions and calculated reference volume.
Add smaller objects, concave objects, glass and shiny surfaces as separate
stress cases.

For each reference, collect at least 30 stationary frames and ten independent
placements/orientations. Check length scale and table flatness first. Then
report dimension bias, repeatability, median/p95 absolute relative volume
error, invalid-depth fraction, boundary sensitivity and unsuccessful
measurement rate. Repeated frames are not independent object placements.

An illustrative height-field relation is `relative volume error ≈ height
error / object height`. A hypothetical 5 mm height bias on a 100 mm object
already contributes about 5% error, before footprint errors. This is a
sensitivity example, not a claim of 5 mm sensor accuracy.

Proposed initial qualification goals for the restricted matte-reference set:
median absolute volume error ≤10%, p95 ≤20%, and explicit rejection when
coverage or shape-fit checks fail. These are experimental goals, not promised
performance. Report errors and acceptance coverage together; revise the
supported size/material domain based on results. Do not present repeatability
alone as accuracy or label an uncalibrated bootstrap interval as total error.

## Benchmark plan and verifiable claims

Use the same private synchronized RGB/depth/IR bundles across candidates.
Start with approximately 100 varied scenes, manually labeled for target
objects, and masks for a representative subset. Use at least 20 distinct
physical objects per material for a pilot, varied across sessions and angles;
broaden this before claiming general material accuracy.

| Claim under evaluation | Evidence required |
|---|---|
| Larger detector improves this deployment | Paired object recall/precision and mask quality on held-out scenes, with per-category failures |
| Material model is useful | Held-out precision, recall and abstention at frozen thresholds; results by visible material and crop policy |
| Artifacts faithfully execute their source models | Preprocessing, boxes/masks and normalized-embedding parity; hashes and immutable revisions |
| A configuration is usable on this NUC | End-to-end p50/p95 latency, input/result age, RSS, CPU use and a 15-minute sustained run |
| Live camera and console remain responsive | Concurrent capture rate, 3D frame rate and API response latency compared with the current baseline |
| Depth dimensions/volume meet the stated domain | Reference-object bias/error and rejection rate across independent placements |
| Multi-view volume represents the scan | Pose checks, surface coverage, watertight/orientation checks, closure metadata and voxel-size sensitivity |
| Experimental workers preserve operator behavior | Independent pause semantics, bounded queues and no saved observations per inference |

Measure detectors alone first, then each material encoder on fixed crops, then
the combined pipeline. Time mask decoding, image preparation, geometry and
serialization as part of end-to-end costs. Report cold startup separately.
Thread counts, model sizes, runtimes and precision must be recorded.
Use at least five warm-ups and 30 samples per scene/configuration, followed by
the sustained run to reveal contention and thermal effects.

After a correct FP32 baseline, investigate GPU execution and representative
INT8 calibration separately. Quantization must preserve detector masks and
material decisions within measured tolerances; it does not repair an
unsupported graph. Compare any GPU or quantized candidate on the same test
split. No GPU speed or accuracy improvement is assumed.

Record geometric method, units, source frames, mask identity, camera calibration,
support plane, model parameters, valid coverage, fit residuals, assumed/filled
regions and calibration-derived error information. Store volume in m³;
display cm³ or mL with a clear distinction between external volume and capacity.
Results that cannot meet a method's preconditions remain unavailable.

Implementation order after this research:

1. Decode and qualify the existing masks; build the fixed evaluation set.
2. Export large/640 and the two first-priority SigLIP2 variants with pinned
   artifacts; compare quality and sustained NUC performance.
3. Add on-demand shape/bounds/height-field measurements and qualify reference
   objects using full-quality depth.
4. Investigate Iris Xe acceleration and quantization; test the heavier
   second-stage candidates only when their quality tradeoff justifies it.
5. Add a separate multi-view scan workflow if arbitrary-object external volume
   is needed. A measured desktop GPU comparison can supplement the NUC results
   once that machine's GPU and memory are inventoried.

## Sources and reproducibility

Primary sources were retrieved on 2026-10-28. Hardware inspection, source
snapshots with hashes, and the official SigLIP2 configurations are retained
locally under `~/.local/state/bolt-rgb/research-2026-10-28/`. No new weights,
driver changes, camera captures, service changes or cloud calls were made for
this research. The prior local measurements remain in `local-vision.md`.

1. Intel compute runtime, supported platforms:
   https://github.com/intel/compute-runtime
2. OpenVINO GPU installation/configuration:
   https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html
3. Ultralytics export contracts:
   https://docs.ultralytics.com/modes/export/
4. Ultralytics YOLOE models and prompt-free LVIS results:
   https://docs.ultralytics.com/models/yoloe/
5. Grounding DINO authors' implementation:
   https://github.com/IDEA-Research/GroundingDINO
6. Google SigLIP2 checkpoints and results:
   https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md
7. Google official configurations, accessed revisions:
   - base/224: `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`
   - base/384: `f775b65a79762255128c981547af89addcfe0f88`
   - large/256: `787800c8990e6f058423089178e718139608408c`
   - large/384: `1b426889ea62b5a72bf9839009a1b184bfc9c178`
   - so400m/256: `e8708ab72d125807e45b36fb7d4e0aacbb59f379`

   The complete model names appear in the material table. Each configuration
   is available at
   `https://huggingface.co/google/{model}/resolve/{revision}/config.json`.
8. MINC authors' dataset and paper:
   https://opensurfaces.cs.cornell.edu/publications/minc/
9. Orbbec Femto Bolt specifications and mode comparison:
   https://www.orbbec.com/products/tof-camera/femto-bolt/
10. Orbbec SDK coordinate-transform example:
    https://github.com/orbbec/OrbbecSDK_v2/blob/main/examples/3.advanced.coordinate_transform/coordinate_transform.cpp
11. Open3D CPU scalable TSDF implementation:
    https://github.com/isl-org/Open3D/blob/main/cpp/open3d/pipelines/integration/ScalableTSDFVolume.cpp
12. Open3D mesh checks and volume implementation:
    https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/TriangleMesh.cpp
