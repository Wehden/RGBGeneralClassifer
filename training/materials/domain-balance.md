# Material loss weighting ablations

The trainer defaults to the original `weight / sqrt(class count)` weighting.
That preserves the original training replay; it does not give the camera
domain an explicit share of optimization.

Use these options with the same frozen dataset, seed, epochs and validation
selection rule to isolate the effect of weighting:

```text
--weighting legacy
--weighting domain-balanced --camera-material-mass .2 --camera-rejection-mass .1
--weighting domain-balanced --camera-material-mass .4 --camera-rejection-mass .1
--weighting public-only
```

In balanced mode, the remaining loss mass belongs to annotated MINC examples.
Each cohort distributes its mass equally across the material classes present
in its training rows; within a class, teacher-quality weights determine each
example's contribution. Validation/test rows never determine these statistics.
A positive cohort mass without training support is an error. Zero-mass rows
are excluded from training and accepted-class support counts.

Reports and export manifests retain requested and effective domain/class
masses. Ranking metrics additionally report top-three, class-macro and
approximate-group agreement. Group IDs remain approximate appearance groups,
not verified identities of physical objects.
Each training run exports both its frozen-encoder head and fine-tuned encoder
variants, so these four runs also provide a bounded comparison of head training
versus encoder fine-tuning.

`--minimum-camera-accepted-references 10` requires at least ten accepted,
referenced camera validation examples and at least 85% agreement before the
artifact becomes qualified for acceptance. It can fail closed even when
public references qualify. The default zero retains the original calibration
rule for baseline replay.

The checkpoint objective remains 70% public and 30% camera top-choice agreement
when camera validation has at least ten target references; otherwise it uses
public agreement. A reviewed dataset with fewer camera references must report
that fallback or explicitly preregister a different
`--minimum-camera-validation-targets` (default 10). Label review and weighting
changes belong in separate ablations. Do not choose runs using the historical
test, whose errors have already been inspected.
The run records the camera validation reference count and the actual public
and camera objective weights to make this fallback visible.

## CPU head screening

Use `--stage head` to run the same 401-step frozen-encoder classifier training,
initialization, duplicate purge, weighting and checkpoint selection without
loading or fine-tuning the encoder. The cached feature files and pinned feature
manifest must already be present in the output directory. Only validation
predictions are computed for evaluation; historical test predictions and
metrics are absent from this stage.

The stage writes `head-state.pt` (a CPU classifier state dictionary),
`head-results.json` (validation metrics, calibration, classifier checksum and
dataset/feature/training provenance), and `head-validation-logits.npz` with
matching `head-validation-rows.json`. These files support a later export after
verification; the head stage does not create or deploy an inference artifact.
The normal `--stage train` still trains and exports both variants.

## Preregistered review exclusions

`--review-manifest PATH` applies training exclusions in memory without editing
teacher labels, dataset files or heldout splits. The JSON manifest must contain:

```json
{
  "schema_version": 1,
  "labels_sha256": "<SHA256 of original teacher-labels.json>",
  "dataset_sha256": "<SHA256 of original dataset.json>",
  "excluded_training_ids": ["<original training sample ID>"]
}
```

Unknown IDs, duplicate exclusions, heldout exclusions and mismatched original
hashes are errors. Optional `source_labels_sha256`/`source_dataset_sha256`
aliases must match; when an `actions` list is present, its `exclude` actions
must match the exclusion list exactly. Validation `flag` actions remain
diagnostic. The manifest checksum and exact exclusions enter run provenance.
Review exclusions never become new heldout references for the duplicate purge.
