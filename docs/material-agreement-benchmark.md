# Blinded material agreement on the detector crops

2026-10-28. This extends the
[YOLOE box-agreement experiment](box-agreement-benchmark.md) to SigLIP2
base/224, base/384 and large/256. Recognition agreement and abstention are
the metrics; execution time is not the ranking criterion.

## Results

All three encoders processed the same 109 distinct RGB crop images that
Nova reviewed. All crops meet the production minimum of 16 pixels on each
original side. These crops represent 110 predictions across the three
detectors; an identical rectangle was deduplicated.

Dedicated, blinded Nova 2 Lite material reviews supplied 20 stable dominant
material references within the 12 target categories. The encoders were
compared against exactly the same reference subset.

| SigLIP2 encoder | Top-choice agreement with stable dominant references | Top choice matches any jointly supported material | Accepted labels, all crops |
|---|---:|---:|---:|
| Base/224 | 1/20 — **5.0%** | 1/34 — 2.9% | 0/109 — 0% |
| Base/384 | 2/20 — **10.0%** | 2/34 — 5.9% | 0/109 — 0% |
| Large/256 | 2/20 — **10.0%** | 2/34 — 5.9% | 0/109 — 0% |

Top choice is measured before gating and includes rejection candidates.
`unknown` does not receive credit against a supported target material.
Accepted-label agreement is **undefined**, because no prediction passed
the existing thresholds. Do not interpret abstention as an accepted wrong
label, or the table as accuracy against human ground truth.

Most ungated predictions were already `unknown`:

| Encoder | `unknown` wins among all 109 crops | Matching dominant references |
|---|---:|---|
| Base/224 | 103/109 — 94.5% | 1 fabric |
| Base/384 | 96/109 — 88.1% | 1 fabric, 1 plastic |
| Large/256 | 97/109 — 89.0% | 2 plastic |

The encoders and their original prompts were unchanged. The acceptance
policy remains similarity ≥0.20, margin ≥0.02 and a non-rejection winner.
Neither threshold tuning nor model fine-tuning was performed on this pilot.

These results do not justify adopting a larger encoder. They support
testing a supervised material head and validation-based calibration on
human-checked material crops. The
[fine-tuning plan](local-model-finetuning.md) describes that path.

## Reference coverage and reviewer limitations

The first object-focused Nova experiment included material fields but
produced only two stable labels in the target vocabulary. A supplementary
material-focused experiment used the same scene and crop pixels, with
two further full sweeps. It did not include SigLIP2 predictions or scores.

| Reference measure | Result |
|---|---:|
| Total distinct crops | 109 |
| Valid material reviews in both passes | 55/109 — 50.5% |
| Stable dominant target material | 20/109 — 18.3% |
| Stable out-of-scope dominant material | 1 human-skin crop |
| At least one target material supported in both passes | 34/109 — 31.2% |

The 20 dominant target references were six fabric, one leather, four paper
and nine plastic. This is a small, imbalanced subset of the material
vocabulary. No general per-material qualification follows from this run.

All 56 Bedrock requests returned responses. **23 responses failed the
strict certainty/evidence schema**, so their whole batches remain missing
references. A frequent inconsistency was reporting a rejection material
as clear while setting its dominant label to null; another was reporting
mixed materials with only one material listed. The protocol requires
consistent certainty, supporting materials and dominant labels.

No inconsistent response was repaired, retried or counted as confirmation.
The strict batch policy also excludes otherwise valid rows in an invalid
batch. This limits reference coverage and can affect which materials
appear in the evaluable subset. The same subset is used for all encoders.

The remaining reviewer judgments are still weak labels. Some supporting
explanations were questionable, including inferring paper from writing
on a whiteboard and plastic from smooth ceiling tiles. Repeated agreement
does not verify composition. The previous object pilot also showed stable
hallucinations and confusion over small objects.

Human review is needed before these labels can become training targets or
a deployment-quality test set. Keep uncertain cases uncertain. Printed
depictions, mixed objects, ambiguous coatings and synthetic lookalikes
need explicit annotation rules.

## Evaluation procedure

The material classifier consumes the exact unannotated JPEG evidence
provided to Nova, including the prior experiment's maximum 512 px crop
size. All original pixels came from the two saved RGB camera frames;
depth and IR are not inputs to this material comparison.

The scene annotations, crop bytes, crop identities and ordering are
identical to the object experiment. Only the review prompt and output
schema change. Each request contains up to four numbered rectangles,
full scene context and the matching crops. Detector names, detector
labels, material predictions and the other pass are hidden.

Nova reports visibly supported materials and surface cues. A clear
assessment requires a supported dominant material. Mixed material
assessments retain several materials without forcing a single label;
uncertain appearance remains unlabeled. Human skin and food are explicit
rejection categories rather than target materials.

The main metric compares an encoder's ungated top choice against the
same target dominant material returned clearly in both passes. The second
metric is a separate diagnostic: whether the top choice belongs to the
intersection of materials visibly supported by both passes. It is not
dominant-material accuracy.

Accepted-label coverage uses all 109 crops. Conditional agreement of
accepted labels uses only accepted predictions with references. Missing,
uncertain, mixed or out-of-scope dominant labels are not invented ground
truth, and are not silently counted as correct.

Each encoder ran in its own CPU process using the existing checked
OpenVINO artifacts, two inference threads and unchanged preprocessing.
The comparison joins predictions and reviews by unique crop identity and
checks image hashes. It does not merge nearby boxes or include duplicate
rectangles as independent material evidence.

## Labeled control and training implications

The previous 200-image public MINC pilot remains the human/category-labeled
material control. Its ungated top-choice accuracy was 35.0% for base/224,
40.5% for base/384 and 24.0% for large/256; accepted-label coverage was zero
for every encoder. Those texture/material images are not deployment camera
crops. They must not be pooled with Nova weak labels as one accuracy metric.

The current camera outputs are dominated by the rejection prompt, even
before acceptance gating. A learned material head on frozen SigLIP2
features is a useful next baseline, with a separate rejection policy
calibrated on validation data. It does not require updating the encoder.
Full encoder fine-tuning is another option if a held-out camera dataset
shows the frozen features are insufficient.

The test set must be split by physical object and capture session. Crops
from these two frames are correlated. Neither the small reference subset
nor Nova repeatability can establish generalization.

## Reproduction and private receipts

Results are under
`~/.local/state/bolt-rgb/material-agreement-2026-10-28/`:
the matched crop dataset, three encoder receipts, dedicated Nova manifests
and raw request receipts, and `paired-summary.json`. Raw images and reviews
are excluded from the public repository.

Run the review CLI with `--review-kind materials` and a separate output
directory. Add `--invoke-bedrock` to perform missing calls. The model is
`us.amazon.nova-2-lite-v1:0`, AWS profile `default`, region `us-west-2`,
temperature zero and disabled reasoning. Completed or interrupted recorded
attempts are never replayed on resume.

The dedicated material reviews recorded 88,334 input tokens and 17,716
output tokens, including schema-invalid responses. The prompt/scoring
implementation hash and image hashes are preserved in the manifest.
Separate sweeps of the same reviewer measure repeatability, not independent
human annotations.

The offline join is reproducible with:

```bash
material_dir="$HOME/.local/state/bolt-rgb/material-agreement-2026-10-28"
.venv/bin/python scripts/summarize-material-agreement.py \
  --reviews "$material_dir/nova" \
  --receipts "$material_dir/siglip2-base-224.json" \
             "$material_dir/siglip2-base-384.json" \
             "$material_dir/siglip2-large-256.json" \
  --output "$material_dir/paired-summary.json"
```

This command makes no cloud calls. The operator archive is not used.
The complete Python suite passed: **266 tests**, including blinding, identical
image evidence across protocols, strict material certainty, mixed-material
handling, rejection categories, crop pairing and undefined accepted agreement.
