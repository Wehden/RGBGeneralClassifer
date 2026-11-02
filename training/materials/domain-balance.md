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
