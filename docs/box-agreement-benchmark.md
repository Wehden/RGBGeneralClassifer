# Blinded Nova 2 Lite agreement with detector boxes

2026-10-28; small, medium and large YOLOE26 prompt-free models at 640 px.
This follows the [local vision pilot](local-vision-volume-benchmarks.md),
using recognition agreement as the selection metric. Execution cost is a
separate feasibility check.

## Results

Nova 2 Lite reviewed every detector box twice, receiving actual RGB images.
The main result requires **both passes to confirm the predicted object label
and judge the box a good fit**. The denominator includes every detection,
including uncertain reviews.

| Detector | Detections | Label confirmed twice | Good fit twice | Label + good fit twice |
|---|---:|---:|---:|---:|
| Small | 32 | 10/32 — 31.3% | 28/32 — 87.5% | **9/32 — 28.1%** |
| Medium | 43 | 12/43 — 27.9% | 39/43 — 90.7% | **12/43 — 27.9%** |
| Large | 35 | 16/35 — 45.7% | 32/35 — 91.4% | **15/35 — 42.9%** |

Large has the highest aggregate agreement in this pilot. The dominant
failure is the label, rather than the box fit. Medium's additional
detections did not translate into a higher confirmed fraction. These
results favor testing large next; they do not establish general accuracy
or justify choosing a model from execution speed alone.

The two frames come from different capture occasions in the same office.
The aggregate advantage is driven by the fresh frame:

| Capture occasion | Small: label + fit | Medium: label + fit | Large: label + fit |
|---|---:|---:|---:|
| Previous office frame | 4/15 — 26.7% | 6/18 — 33.3% | 4/14 — 28.6% |
| Fresh office frame | 5/17 — 29.4% | 6/25 — 24.0% | 11/21 — 52.4% |

One small-model crop was uncertain in one pass. The other 109 predictions
were identifiable in both passes. Restricting the denominator to identifiable
predictions changes small's joint result to 9/31 — 29.0%; medium and large
remain unchanged.

## Reviewer stability

There were 109 distinct crops for 110 detector predictions. One exactly
identical crop was shared between detectors and reviewed once per pass,
with both predictions scored against that evidence.

| Nova self-agreement | Result |
|---|---:|
| Same canonical primary object label, identifiable in both passes | 94/108 — 87.0% |
| Same box-fit category, all crops | 103/109 — 94.5% |
| Clear, identical dominant material in both passes | 5/109 — 4.6% |

Primary-label self-agreement uses the observed main labels and the frozen
alias table; it does not use the reviewer's additional equivalent names.
Box-fit self-agreement includes matching unfavorable categories.

| Detector | Pass 1: label + fit | Pass 2: label + fit | Both passes: label + fit |
|---|---:|---:|---:|
| Small | 10/32 — 31.3% | 10/32 — 31.3% | 9/32 — 28.1% |
| Medium | 15/43 — 34.9% | 12/43 — 27.9% | 12/43 — 27.9% |
| Large | 17/35 — 48.6% | 15/35 — 42.9% | 15/35 — 42.9% |

All 56 requests completed and passed schema validation, without retries
or model fallback. Recorded usage was 89,118 input tokens and 19,198 output
tokens. The model was `us.amazon.nova-2-lite-v1:0`, through Bedrock in
`us-west-2`, using AWS profile `default`, temperature zero and disabled
reasoning.

These are separate calls to the same reviewer with identical inputs.
They measure repeatability, not agreement between statistically independent
annotators. Requiring two confirmations reduces unstable judgments; it
cannot remove a stable hallucination.

## Protocol and matching

The experiment uses the original and first fresh frame from the prior
detector receipts. Adjacent fresh frames were excluded to avoid counting
near-identical video frames as additional scenes. All three detectors use
the same underlying RGB frames, 640 px input, 4,585-label prompt-free
vocabulary, FP32 ONNX exports and 0.45 detection confidence threshold.
Depth and IR are not reviewer inputs.

For each detector box, the runner calculates the production crop bounds:
floor the lower coordinates, ceil the upper coordinates, and clip to the
frame. Only identical integer rectangles on the same frame are deduplicated.
No approximate overlap clustering changes the evidence being evaluated.

Cases are shuffled with a fixed seed and presented in groups of at most
four. Each request contains an annotated full-resolution scene with
numbered rectangles and the corresponding unannotated RGB crops, reduced
to at most 512 px on their longest side without upscaling. There are no
detector names, predicted labels, confidence scores or previous reviews in
the request. Crops are JPEG-encoded from the original unannotated pixels.
Two full sweeps use identical prompts and images.

Nova returns an ordinary physical-object or surface label, equivalent
names, visibility, a box-fit category and a visual explanation for each
rectangle. Fit categories are `good`, `loose`, `partial`, `multiple`,
`background` and `uncertain`. Frame-edge clipping alone does not fail a
box; an identifiable partly occluded object can still pass.

Label matching uses normalized spelling, a conservative alias table fixed
before invocation, and Nova's blinded equivalent names. For example,
`office chair` and `chair` match. Occupations do not become `person`;
`podium` does not become `desk`; `television` does not become `monitor`.
The exact policy and scoring implementation hash are stored in the
manifest and private summary:

```text
3ad75b9d93c4c98038ecb0239cee3d8dded66f6e131911668d90929ae4a5c1ab
```

A supplementary spelling-only check, still including Nova's equivalent
names but removing the fixed alias expansion, confirmed labels for
9/32 small, 10/43 medium and 16/35 large predictions. It is a label-only
diagnostic, not the joint fit score.

Malformed, missing or interrupted responses remain unconfirmed. Receipts
are written before invocation, and resume never replays a recorded
attempt, including a failed or interrupted one.

## Interpretation and limits

This is a candidate-box agreement proxy, not measured precision against
human ground truth. It does not measure missed objects, recall, segmentation
quality or numerical bounding-box IoU against independently predicted boxes.
Fit is Nova's categorical assessment of the supplied rectangle. Boxes from
the same office and repeated instances of an object are correlated; 110
predictions are not 110 independent scenes.

Examples of systematic detector/reviewer disagreement include small's
`jazz artist` and medium's `folk artist` on a person, medium's `sound` on
headphones, and a large-model `podium` box reviewed as a monitor. These
specialized or abstract detector labels do not receive person/object credit.
Several `plectrum` detections were also reviewed as other objects.

The reviewer has visible weaknesses. It called a small red disc-shaped
accessory on a display a bottle in **both** passes. It identified the
depicted food in a printed image rather than distinguishing the printed
object. One crop changed from chair to backpack. Some returned equivalent
names were overly broad, including cup/bottle alternatives. Such aliases
can affect detector agreement. The protocol was not adjusted after seeing
these results; the private receipts preserve the original judgments.

The five stable material labels were three human-skin crops, one paper crop
and one glass crop. Only two fall within the deployed SigLIP2 material
vocabulary. This is insufficient to rank the material encoders; their
[labeled MINC pilot](local-vision-volume-benchmarks.md#material-execution-and-labeled-pilot)
remains the quantitative comparison.

Large is the strongest candidate in the combined agreement table. A broader
set of scenes with human-checked labels and boxes is needed to validate that
advantage and the Nova reviewer, especially for small objects. The runner
can evaluate new saved receipts using the same frozen procedure.

## Evidence and reproduction

Private evidence is stored under
`~/.local/state/bolt-rgb/box-agreement-2026-10-28/`: source and image hashes,
`manifest.json`, actual JPEG inputs, per-request raw responses and request
IDs, `cases.json`, and `summary.json` with aggregate, per-scene and per-pass
counts. Camera images and raw reviews are excluded from the public repository.

To reproduce against the saved detector receipts:

```bash
qualification_dir="$HOME/.local/state/bolt-rgb/qualification-2026-10-28"
review_dir="$HOME/.local/state/bolt-rgb/box-agreement-2026-10-28"
.venv/bin/python scripts/benchmark-box-agreement.py \
  --receipts "$qualification_dir/yoloe-26s-pf-640.json" \
             "$qualification_dir/yoloe-26m-pf-640.json" \
             "$qualification_dir/yoloe-26l-pf-640.json" \
  --output "$review_dir" --scene-indices 0 1
```

Without `--invoke-bedrock`, this prepares or verifies evidence and summarizes
existing receipts without cloud calls. Add that flag to perform missing
reviews; use a new output directory for a fresh experiment. Model, profile
and region are explicit CLI options. The default cap is 64 new calls per run;
this experiment needed 56 total.

The runner uses saved frames and writes its own qualification directory.
The operator archive and live capture are not involved.
Regression checks cover uncertainty and missing-review handling, conservative
label matching, exact box identities, material abstention, frame identity,
image hashing, blinding, identical evidence across passes, and safe resume.
The complete Python suite passed: **258 tests**.
