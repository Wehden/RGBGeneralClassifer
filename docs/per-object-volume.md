# Per-object volume in the local console

The local Femto Bolt runner attaches a `volume` reading to each YOLOE
detection. RGB-only V4L2 cameras omit that field and skip mask/geometry
processing. The comparison and records tables hide their volume column
when the displayed captures have no volume readings. Old observations
remain readable without a volume field.

## Meaning of the estimates

The detector uses its existing segmentation head in the same inference
pass. Each RGB instance mask selects independent native depth pixels using
that frame's calibration and depth-to-color projection. Quantitative points
remain in native depth-camera coordinates. Bounding-box background and
resampled registered pixels are not used as independent geometry samples.

General objects receive a trimmed, PCA-aligned **visible-point box proxy**.
The proxy describes the observed cloud; it cannot establish unseen backs,
undersides, voids or full external volume. Near-planar surfaces receive
`visible_surface_only` with a null estimate rather than a zero-volume object.

Supported cylindrical labels can receive a **conditional external cylinder
estimate**, using observed height and silhouette width. A well-supported
ceiling, floor, table or desk plane supplies an assumed vertical axis.
Its orientation is a shape/scene assumption, not measured gravity.
The cylinder requires sufficient native samples, plausible extents,
consistent depth span, and a mask that does not reach the image boundary.
If closure is unsupported, a usable visible-point proxy can remain,
with the cylinder rejection reason retained separately.

Every result carries source frame/session/calibration identity, method,
sample counts, dimensions in metres, volume in m³, warnings and
`physical_accuracy: "unverified"`. The UI displays mL or L, with `(cyl)`
next to readings that use a successful cylinder estimate. External volume
does not establish vessel capacity or material composition.
No volumes are summed across detections: masks can overlap, detections can
describe parts of the same object, and single views are incomplete.

The estimator records a second result with three pixels removed from each
mask boundary. This is a mask-sensitivity scenario, not a confidence
interval or an accuracy bound. Factory rotation warnings remain visible.
Registered-depth mask coverage is a separate diagnostic.

## Availability and capture-camera support

| Capture case | Behavior |
| --- | --- |
| RGB-only camera | No detection `volume` field, no volume overlay, no volume column for an RGB-only result page |
| Supported depth camera, missing native depth | `no_native_depth`, null estimate |
| No usable segmentation mask | `no_instance_mask`, `empty_instance_mask` or `invalid_instance_mask`, null estimate |
| Too few native samples | `insufficient_depth`, null estimate |
| Only out-of-mode returns | `outside_operating_range`, null estimate |
| Invalid calibration/depth | `invalid_calibration_or_depth`, null estimate; object detection continues |
| Unsupported native profile | `unsupported_depth_profile`, null estimate; no range is guessed |
| Partial visible cloud | Visible box proxy with partial-view semantics |
| Supported cylindrical closure | Conditional cylinder estimate with listed assumptions |

The current local camera adapters are Femto Bolt (`orbbec`) and RGB-only
V4L2. Femto Bolt native profile dimensions select the operating-range
filter: 640×576 uses 0.5–3.86 m, 320×288 uses 0.5–5.46 m, 512×512 uses
0.25–2.88 m, and 1024×1024 uses 0.25–2.21 m. Zero and uint16 saturation
sentinels are excluded. Nonzero far returns are recorded diagnostically,
but cannot produce an estimate outside that active mode's range.
Source: Orbbec's *Femto Bolt hardware specifications*, checked 2026-11-02:
`https://www.orbbec.com/documentation/femto-bolt-hardware-specifications/`.

## API, archive and UI

`GET /api/camera/detections` includes the reading alongside each box and a
frame-level `metrology` receipt with runtime and the candidate vertical
reference. All geometry is computed before publishing the completed
detection frame. Numeric arrays/masks remain transient and are not exposed
as JSON.

The live RGB overlay, archived box labels, item details, model-comparison
rows and records page show the same readings. Item details expand to
dimensions, sample counts, shape assumptions, mask sensitivity and warnings.
Comparison readings retain the source frame alongside the original object
and material scores.

Captured observations preserve these values. Records JSON includes `volume`
only when the detection has it. Both page downloads and full Arrow/Parquet
exports contain a nullable typed `volume` struct plus `volume_json` with the
complete receipt. Full export schema version is 10. RGB-only rows have null
export values, never synthetic zeros.

There is no new listener or camera owner: this runs inside the existing local
service and its singleton capture pipeline. Cloud labeling is not involved.

## Repeatable verification

Run the geometry, capability, persistence and renderer checks:

```bash
.venv/bin/python -m pytest -q
node --test tests/js/*.test.cjs
```

Replay retained frame bundles through the same detector/mask/volume path.
Use a new private output directory; receipts and annotated scene images
contain capture evidence and should remain private.

```bash
.venv/bin/python scripts/benchmark-object-volume.py \
  --frames /private/frame-01.npz /private/frame-02.npz /private/frame-03.npz \
  --model-dir "$HOME/.cache/bolt-rgb/models/yoloe-26s-pf-640" \
  --output-dir /private/object-volume-small
```

Repeat with the installed large/640 detector and another output directory
to compare mask quality without changing the console's detector choice.
Receipts record bundle/model/calibration hashes, frame identity,
inference-plus-mask time, geometry time, estimates and rejection reasons.

The 2026-11-02 NUC run completed on all three retained bundles with both
small and large detectors. Small emitted 15–17 boxes per frame; geometry
took 265–318 ms. Large emitted 18–21 boxes; geometry took 262–279 ms.
The large detector reproduced the prior cup cylinder estimates:
0.388, 0.388 and 0.446 L. Eroded masks produced 0.268, 0.264 and 0.296 L.
The usable bottle case reproduced 0.385 L, versus 0.155 L with erosion.
The small detector's corresponding cup/thermos masks lacked sufficient
depth, and returned unavailable readings.

A fresh-camera smoke test completed with frame-qualified readings on all
12 detected boxes. An isolated, capture-paused/no-cloud runner then saved
one manual observation and a Chromium check verified volume details on
all 11 boxes in that later capture and the records column, without page
errors. The production archive/settings were not modified. Raw receipts
and the console screenshot remain in the private dated experiment directory.

These runs establish integration, arithmetic and rejection behavior.
Physical accuracy still requires independently measured objects across
placements, materials, boundaries and partial views. See
[the range/volume experiment](capture-range-volume-experiment.md) and
[the research qualification plan](local-vision-volume-research.md).
