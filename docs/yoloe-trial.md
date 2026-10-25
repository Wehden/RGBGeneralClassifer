# YOLOE-26n trial on Uno Q

The trial runs the official YOLOE-26n prompt-free segmentation checkpoint,
exported to float32 ONNX at 320 px. Its built-in vocabulary contains 4,585 names.
The application consumes its boxes and class scores; it does not publish masks.
The end-to-end graph selects the highest class per proposal, with up to 100
outputs. There is no added spatial NMS pass.

The manifest in `models/yoloe-26n-pf-320/` contains the complete label mapping,
source checkpoint URL/hash, export options and resulting ONNX hash. These
weights are AGPL-3.0, with the upstream license alongside them. The original
EfficientDet artifact and retained Uno Q image remain available for rollback.

## Measurements

The initial comparison used two archived 640x480 camera images, two CPU threads,
two warm-up calls and 12 timed calls per model, with the live detector stopped.
Peak RSS is the benchmark process, measured independently for each model.

| Model | Input | Median inference | p95 inference | Peak RSS |
|---|---:|---:|---:|---:|
| EfficientDet-Lite0 |320|154.25ms|155.19ms|93,860KiB|
| YOLOE detection-only,29 desk prompts |320|212.86ms|214.45ms|108,060KiB|
| YOLOE detection-only,29 desk prompts |640|805.15ms|807.21ms|170,536KiB|
| YOLOE prompt-free segmentation |320|774.44ms|778.26ms|483,076KiB|

The timing comparison used the export's default multiple-class proposal selection.
The deployed export uses `agnostic_nms=True`, which selects one class per proposal
inside its end-to-end head. Its separate live measurements are recorded in the
verification receipt. Both configurations are recorded rather than treating
these timing runs as identical artifacts.

The desk profile used these 29 names: person, computer monitor, television,
laptop, keyboard, computer mouse, headphones, fan, cup, mug, bottle, beverage
can, glass, plate, bowl, book, paper, cardboard box, plastic bag, plastic film,
food container, pen, cable, charger, phone, scissors, tape, desk and chair.
Those prompts are fixed in the exported desk graph. Prompt-free mode uses its
released vocabulary and needs no text encoder on the Uno Q.

Raw paired readings and image hashes are retained privately with the trial
results. Examples of prompt-free readings at the configured 0.39 threshold
included `playing card`, `bureau`, `jack`, `crt screen` on the wide desk image,
and `cup`, `keyboard`, `office desk` on the second image.

## Runtime and records

`DETECTOR_BACKEND=yoloe` chooses the ONNX adapter. `YOLOE_MODEL` and
`YOLOE_MANIFEST` identify the pinned graph and vocabulary. ONNX Runtime 1.24.4
uses its CPU provider, two inference threads and one inter-op thread.
`MAX_FPS=0` leaves capture uncapped; throughput follows actual processing time.
Snapshots remain scheduled every 15 seconds when detections exist.

Each observation records its model hash, runtime/version, input size, prompt
mode, export options and vocabulary/manifest hashes. The full vocabulary lives
in the versioned manifest instead of being duplicated in every observation.
Class IDs are scoped to that model's vocabulary. Schema6 complete exports add
`detector_config_json` to observations and `initial_class_id` to detections.
Paged downloads also retain initial model/hash, configuration and class ID.
Older observations remain readable with null configuration fields.

## Build and rollback

Build the baseline board image first, then the YOLOE extension:

```sh
docker build -f board/Dockerfile -t rgb-classifier:board-base .
docker build -f board/Dockerfile.yoloe -t rgb-classifier:unoq-yoloe26n-20261025-v2 .
```

The extension contains the exported graph and ONNX Runtime. It installs no
PyTorch or text encoder. The App Lab brick owns port 8080 and the camera.
On the current board, the previous compose file is retained at
`/home/arduino/rgb-classifier/brick_compose.v7-rollback.yaml`. Restoring that
file to the app's brick directory and restarting the app restores the v7
EfficientDet image and its prior capture settings. It does not erase records.

For export changes, use an isolated Python 3.12 environment. The tested versions
are torch 2.9.1+cpu, torchvision 0.24.1+cpu, ultralytics 8.4.188, onnx 1.20.1,
onnxslim 0.1.82 and onnxruntime 1.24.4. `scripts/export-yoloe.py` verifies the
checkpoint hash, exports the graph and updates its manifest hash. Export
metadata can change the artifact's byte hash; keep each generated manifest
with its graph. No model download occurs during board startup.

## 9B VLM pool

RGB now targets `http://ollama-vlm-pool.ollama.svc.cluster.local:11434` with
`qwen3.5:9b-q4_K_M` and `num_ctx=8192`. Its egress rule selects the `ollama`
namespace and pods labeled `localk8s.io/ollama-vlm-group=qwen9`, rather than
allowing the old 27B host. Pool membership and other consumers are unchanged.

A bounded driver probe from RGB's namespace returned HTTP 200 in 6,934.75 ms.
It read the archived monitor box as `computer monitor` 0.95, with glass 0.98 and
plastic 0.92. The private validation transcript retains the prompt, schema,
raw response, image identity, options and timings. The probe did not advance
the analysis queue or replace the archived candidate. Background VLM work
remained paused. `/api/tags` is supported by the pool; `/api/version` returned
404. The driver uses tags and chat, so version routing is not required.
