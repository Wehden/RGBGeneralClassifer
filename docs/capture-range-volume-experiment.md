# Current capture: range and conditional object volume

2026-10-29, NUC13 / 32 GB / Femto Bolt. Three synchronized RGB/native-depth/IR
bundles were acquired at 15:01:01–15:01:03 UTC. The desktop service was closed,
so a temporary camera owner acquired and released the singleton directly.
No observation archive, settings or cloud-review queue was modified.

The offline detector was the already-installed YOLOE26 large/640 artifact.
This does not change the console's detector selection. Masks and numerical
geometry use each frame's own calibration and retained pixels.

## Full observed range

The previous quantitative helper's default selection was 0.5–3.86 m.
This experiment explicitly selects the entire nonzero, unsaturated native
depth range of each frame. The helper's original default remains available.
Nonzero returns are observations, not a qualification of the sensor at that
distance.

The captured native profile is 640×576, which Orbbec identifies as NFOV
unbinned with a **0.5–3.86 m operating range**. Returns beyond that range
are retained for diagnostics and explicitly marked outside the mode's
specification. They must not be interpreted as validated long-range
measurements. NFOV 2×2 binned trades resolution down to 320×288 for a
published 0.5–5.46 m operating range; that mode was not tested here.
The source is Orbbec's *Femto Bolt hardware specifications*, retrieved
2026-10-29:
`https://www.orbbec.com/documentation/femto-bolt-hardware-specifications/`.

| Frame | Axial minimum / maximum | Axial p01 / p99 | Pixels returning depth |
|---|---:|---:|---:|
| 1 | 0.661 / 7.778 m | 0.683 / 6.574 m | 61.1% |
| 2 | 0.660 / 8.275 m | 0.682 / 6.590 m | 61.2% |
| 3 | 0.679 / 9.151 m | 0.718 / 5.516 m | 62.0% |

Across the frames, raw observed axial depth spans **0.660–9.151 m**.
The extreme far return is unstable; central percentiles are more useful
for describing this capture. Subject motion/occlusion also changes which
background is observed. Approximately 38–39% of native pixels lack depth.
Approximately 11% of the returning samples are outside the current mode's
specified range.

Axial depth is the coordinate along the depth camera's optical axis.
Calibrated Euclidean camera-to-point distances span approximately
**0.700–9.250 m** across these frames. Their per-frame p99 values are
6.709, 6.717 and 5.974 m. These remain unverified physical measurements.
The earlier SDK parity result does not establish far-range accuracy.

## Object estimates

Native points are selected by the RGB instance masks through the factory
transform, retaining the earlier calibration warning. The SDK-registered
depth image supplies a separate mask-coverage diagnostic; its resampled
pixels are not counted as independent native measurements.

The ceiling mask supplies a candidate plane. Its normal is assumed to be
vertical because the ceiling is assumed horizontal; this is not a measured
gravity vector.
Only points within the current mode's specified range contribute to this
candidate plane; the final fits have 99.0–99.7% inliers within 10 mm.
An unsupported ceiling plane is recorded as a rejected candidate, and the
runner continues to any later ceiling candidate. If no vertical reference
is supported, it still writes range, visible-envelope and annotated-image
receipts for every frame; cylindrical labels report `no_vertical_reference`.
Invalid geometry inputs continue to raise validation errors.
Only explicitly supported cylindrical labels receive a cylinder closure.
Height is measured along the candidate vertical axis, and diameter across
the viewing direction in the horizontal plane. Each uses 2nd–98th percentile
extents to reduce outliers.

| Object / detection | Typical axial depth | Result |
|---|---:|---|
| Paper cup | 1.23 m | Full-mask external cylinder envelope **0.388, 0.388, 0.446 L** |
| Same cup, mask eroded by 3 pixels per boundary | 1.23 m | **0.268, 0.264, 0.296 L** |
| `bottle` detection by the right desk | 2.66 m | **0.385 L** in frame 3; rejected for inconsistent depth span in frames 1–2 |
| Same bottle, eroded mask | 2.66 m | **0.155 L** in frame 3 |
| Thermos | Unavailable | Only two selected native samples per frame and no registered depth in its mask; no estimate |
| Earphone/headphone detection | 0.75–0.85 m | Partial visible box roughly 17–19 × 7–8 × 7 cm; no full-object closure |
| Left backpack detection, frames 1–2 | 1.45 m | Partial visible box roughly 26 × 21 × 10 cm; box proxy 5.5–5.7 L |
| Far chair | 5.28 m raw return | Outside this mode's operating range; no qualified dimensions/volume |
| Whiteboard | 3.05 m | Partial planar surface; thickness/full volume not established |

The cup's full-mask estimate corresponds to approximately 7.9 cm diameter
and 7.8–9.1 cm height. For a tapered cup, the cylinder is a shape assumption;
it does not establish vessel capacity, internal voids or material volume.
The bottle label is also a detector judgment, not an independently verified
object annotation.

Mask sensitivity is substantial. The cup's complete scenario spread is
approximately **0.26–0.45 L** and the bottle's is **0.16–0.39 L** in its
usable frame. These are variations between masks/frames, **not confidence
intervals or verified bounds on physical volume**.

The broad ray-depth span for the bottle in frames 1–2 indicates incompatible
layers within its mask; it is not silently filled or used to infer diameter.
Object labels, background leakage, missing depth, incomplete views and the
unqualified RGB/depth registration remain sources of error.

No total object volume is summed: person/clothing/headphone detections
overlap, furniture views are incomplete, and bounding boxes of visible
points do not enclose complete objects. Their dimensions and envelope
proxies remain explicitly separate from conditional shape estimates.

## Repeat the experiment

Retain immutable bundles through the existing camera frame API or a
temporary singleton camera owner, then run:

```bash
.venv/bin/python scripts/estimate-capture-volume.py \
  --frames /private/frame-01.npz /private/frame-02.npz /private/frame-03.npz \
  --model-dir "$HOME/.cache/bolt-rgb/models/yoloe-26l-pf-640" \
  --output-dir /private/capture-metrology
```

Each JSON receipt records the source hash, frame/calibration identity,
detector hash, raw and robust range, native sample counts, registered-mask
coverage, candidate vertical plane, visible dimensions, conditional volume,
rejection reasons and mask sensitivity. Annotated RGB images remain private.
The original run is under
`~/.local/state/bolt-rgb/capture-volume-2026-10-29/`.

Sixteen geometry/metrology and runner tests pass, including a known cylinder
seen from its front half, invariance to rotation, invalid/saturated depth,
explicit far-range selection, planar surfaces and mixed-depth rejection.
Runner regressions verify complete batch receipts without a supported ceiling,
recovery through a later candidate, and propagation of invalid-input errors.
The live three-frame run exercises the complete offline runner. Source
compilation and whitespace checks pass.

The next physical check should compare a known cup/cylinder and cuboid
against independently measured external dimensions, including deliberate
boundary/occlusion changes. Current output is an exploratory estimate, not
a production volume endpoint.
