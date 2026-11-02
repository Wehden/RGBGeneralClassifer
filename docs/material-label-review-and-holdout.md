# Material label review and fresh holdout protocol

The camera teacher labels describe visible appearance. Agreement between two
VLM passes is a weak reference, not a measurement of physical composition.
Keep that distinction when reporting model agreement and when building a
material accuracy benchmark.

## Review without changing the evaluation question

Freeze the original dataset, teacher labels, crop files, split assignments,
feature caches and evaluation protocol before an experiment. Record their
SHA256 hashes in the private experiment receipt. Never train on original test
examples, change their labels, or choose thresholds or variants using their
results. Existing test results remain historical evidence rather than a fresh
confirmation set after repeated experiments.

Review original training and validation crops alongside their original teacher
evidence. Do not use a student's predicted material to decide which teacher
label to retain. Use these crop flags:

| Flag | Observable reason | Appropriate action |
| --- | --- | --- |
| Conflicting weak references | Approximate views of one surface have contradictory targets | Exclude training targets without replacing them |
| Teacher target mismatch | Described object or surface is absent or ambiguous in the crop | Exclude training target or flag for human review |
| Mixed surfaces | Crop combines padding, frame, skin, screen, or background | Flag; define a surface patch before requesting another label |
| Uncertain composition | Visual cues cannot distinguish glass/plastic, rubber/plastic, or coated surfaces | Retain uncertainty; require independent composition evidence |
| Background dominated | Named target occupies only a small part of a scene crop | Flag; use a target mask or a smaller verified patch |
| Insufficient visible area | Target is tiny, occluded, blurred, saturated, or edge-truncated | Flag or exclude training evidence; do not invent composition |

Preregister a training exclusion list before running the ablation. Disputed
validation references stay in the frozen evaluation, with separate diagnostic
reporting; deleting them would change the calibration/evaluation question.
Adding more exclusions after seeing the experiment's results requires a new
preregistered experiment rather than silently revising the first one.

The private review manifest uses `schema_version: 1`,
`source_dataset_sha256`, `source_labels_sha256`, and `actions`. Each action
contains `id`, original `split`, crop `sha256`, original `group_id`,
`source_consensus_sha256`, `decision` (`exclude` or `flag`), `flags`, and
`note`. The consensus hash is SHA256 of sorted, compact JSON. Optional
`dataset_sha256`, `labels_sha256`, and `excluded_training_ids` aliases must
agree with the canonical fields and actions. No action may assign a replacement
material, weight, or split.

`scripts/apply-material-review.py` checks this provenance and writes a new
derived label document. It allows training exclusions and train/validation
flags, rejects validation exclusions and all test actions, preserves raw
teacher evidence, and refuses to overwrite an existing output:

```sh
python scripts/apply-material-review.py \
  --dataset /private/original/dataset.json \
  --labels /private/original/teacher-labels.json \
  --review /private/review/proposed-review-manifest.json \
  --output /private/review/derived-teacher-labels.json
```

An exclusion clears only the derived training consensus target and weight.
The original documents and test rows stay intact. The exclusion-only training
ablation can instead consume the bound manifest directly; do not substitute a
derived label document into an existing immutable feature cache.

## A holdout that can measure physical material accuracy

New timestamps, session IDs, frame hashes, camera viewpoints, or detector
labels establish new observations. They do not establish different physical
objects. Appearance grouping and session-local tracking are useful overlap
alarms but cannot prove object identity or composition. A no-match against a
sampled crop subset also cannot prove absence from the original footage.

Acquire a dedicated collection of known material surfaces. Use owner-verified
composition, a manufacturer's specification, or a documented reference sample.
Record the evidence source and distinguish exposed coating from bulk substrate.
When composition remains uncertain, use an unknown or mixed evaluation category;
do not turn a confident VLM guess into ground truth.

Before collecting images, register:

1. A permanent `physical_object_id`, independent of camera tracking/session IDs,
   and a `material_surface_id` for each exposed surface. Record whether the same
   object appeared in previous camera footage; uncertain identity cannot qualify
   for a confirmed novel-object holdout.
2. The independent composition reference and its uncertainty, material taxonomy,
   and allowable mapping. Include hard negatives such as painted metal, coated
   wood, transparent plastic, glass, rubber-like foam and mixed assemblies.
3. Separate development and confirmation object lists. Every view, patch and
   session of one physical object belongs to the same split. Keep the confirmation
   list unavailable to training, threshold fitting and experiment selection.
4. Capture conditions: distance, viewpoint, lighting, image/depth calibration,
   crop or mask provenance, foreground coverage, resolution and focus. Record
   RGB/depth/IR assets with hashes where available; validate depth registration
   and coverage before using it as mask evidence.

For an initial balanced collection, aim for at least ten distinct physical
objects per intended supported material class, split five development and five
confirmation objects, with three views per object. This is a pilot target:
five confirmation objects per class leave wide uncertainty. Extra views of
the same object improve view coverage but do not increase the number of
independent objects. Expand class/object coverage before claiming deployment
precision, especially for sparse classes.

Choose architectures, preprocessing, prompts and rejection thresholds using
development objects and the original frozen validation protocol. Evaluate a
preregistered finalist on confirmation objects once. If another round is needed,
acquire another confirmed object-disjoint collection.

## Surface and segmentation experiments

Compare the same labeled surface across these preregistered variants:

| Variant | Question answered | Required safeguard |
| --- | --- | --- |
| Original detector crop | Existing baseline | Keep reference and box provenance |
| Foreground mask with preserved aspect ratio | Does background contamination hurt classification? | Audit mask target, disconnected regions and padding |
| Interior surface patch | Do texture and reflectance cues improve material separation? | Patch must lie on one independently identified surface |
| Multiple surface patches | Does an assembly contain several materials? | Aggregate per surface; do not force one whole-object material |
| RGB/depth foreground patch | Does registered geometry improve target isolation? | Handle holes, reflective surfaces and depth misregistration |

Masks isolate an object rather than necessarily isolating one material. A chair
or headset can contain several surfaces inside a correct mask. Avoid center
patches that silently select background, skin, seams or a different component.
Save mask/patch hashes, foreground fraction and extraction rules before
evaluation so comparisons can be repeated. Report paired changes per object,
not only per crop.

Report top-1 and top-3 agreement with weak camera references separately from
accuracy on verified material references. Include per-class support, confusion,
macro accuracy, object/group aggregate accuracy, accepted precision and coverage,
and unresolved rejection behavior. Fit acceptance gates on enough development
objects from the camera domain; pooled public-dataset examples cannot establish
camera precision. Keep MINC's independently annotated test results separate
from VLM-labeled camera agreement.
