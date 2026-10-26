# YOLOE materials mode

Objects mode runs the existing 4,585-class prompt-free model. Materials mode
runs a second YOLOE-26n graph with 12 baked text prompts: plastic, paper,
metal, film, liquid, glass, cardboard, wood, textile, rubber, ceramic, and foam.
The manifest maps each label to its actual prompt phrase. Switching modes
selects the corresponding model and resets tracking identities. Both graphs
use ONNX Runtime CPU with two threads; inference has no FPS cap.

The materials graph is `models/yoloe-26n-materials-320/model.onnx`, SHA-256
`23960eb842490d643721b1bd14c5c8491798343d91b3320e6511a4d5388a6bcf`.
Its official `yoloe-26n-seg.pt` checkpoint has SHA-256
`1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b`.
`scripts/export-materials.py` pins Ultralytics 8.4.188 and records the checkpoint,
text encoder hash, prompt phrases, export settings, and graph hash. Text
encoding happens off-board; the Uno Q loads the static graph. This uses the
[documented YOLOE prompt export contract](https://docs.ultralytics.com/models/yoloe/).

Local material labels are displayed immediately. Qwen subsequently returns
material readings, per-material confidence, and reasons; accepted readings
replace the live material label when they match the current track. Qwen's
object name stays separately recorded. For material detector inputs its
comparison prompt compares the initial material label with its material read.
The Classifier slider applies to the selected local model; the VLM slider
applies to Qwen material readings.

Recording includes frames with zero accepted detections. Each observation
retains up to 100 scored detector proposals before threshold filtering, so a
zero-box result can be reviewed against its image and configured threshold.
Preview mode does not record these values. Complete Arrow/Parquet exports use
schema 7: observations include proposals and capture-time VLM policy; detection
rows include `initial_label_kind`; material rows distinguish `detector` and
`vlm` providers. `analysis_actions` records manual requests and catch-up batches.
The records material filter also finds initial YOLOE material labels.

## Native board measurement

A six-run test on one archived desk frame produced inference times of 396.04,
372.07, 378.15, 387.31, 389.69, and 362.46 ms. At the existing operator threshold
of 0.39, the graph returned zero accepted boxes. A separate unfiltered read
returned a highest proposal of `rubber`, score 0.2526. This receipt records
execution and model outputs; no reference material annotations were supplied.
Private source images and complete readings are retained outside the repository.
