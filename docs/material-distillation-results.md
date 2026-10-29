# Material student training results — 2026-10-28

The first training run substantially improved material recognition on the
held-out annotated control and agreement with the camera's weak Nova labels.
The fine-tuned model was selected by validation. Camera acceptance remains
sparse, so this is an experimental material-appearance model.

See the [overall plan](material-distillation-plan.md) and
[repeatable process](material-distillation-process.md).

## Inputs and teacher

The available capture source was 144 retained RGB observation frames with
966 boxes across three sessions, rather than a continuous video file.
The archive was read-only throughout the exercise.

The frozen snapshot contains 192 representative crops from 154 approximate
appearance/track groups: 134 train, 29 validation and 29 test. Source image
hashes, frame identities, session assignments and crop bounds are retained.
Three camera training crops were additionally excluded because their frozen
feature similarity to a held-out camera crop was at least 0.985.
An audit found no selected exact duplicates crossing sessions. The repeatable
extractor also has a regression check to preserve the later occurrence when
identical pixels recur in future captures.

Nova 2 Lite (`us.amazon.nova-2-lite-v1:0`) labeled actual annotated scene
images and unannotated crops in two blinded passes through the local
`default` AWS profile. The 96 calls used 224,640 input and 29,551 output
tokens. All 192 crops have two parsed reviews.

| Teacher outcome | Crops |
|---|---:|
| Strong dominant agreement | 168 |
| Moderate dominant agreement | 2 |
| Mixed materials | 6 |
| Disagreement | 12 |
| Uncertain | 4 |

An initial JSON wrapper variant was recovered by deterministic parsing of
the retained response; it was not relabeled or billed again. Requests,
raw responses, image hashes, protocol hash and parser source hash remain
in the private run directory.

The 170 agreed labels comprise 111 human-skin rejection examples, 3 food
rejection examples and 56 target-material examples. These are weak VLM
references, not verified composition.

Add 1,400 annotated MINC images from official split 1: 100 train,
20 validation and 20 test per category across ten categories. All 200
images from the earlier pilot are excluded from every new split.

After uncertainty and duplicate filtering, training uses 1,114 examples:
1,000 MINC and 114 camera crops. Camera training contains 86 skin rejection
examples and 28 target-material examples. Rubber has only two training
examples; cardboard, food and unknown have no hard training examples.
These sparse classes cannot become accepted material identifications.

## Training and selection

Both experiments start from `google/siglip2-base-patch16-384`, revision
`f775b65a79762255128c981547af89addcfe0f88`. The pinned checkpoint's
Transformers implementation is `siglip`, with 768-dimensional normalized
pooled image features.

Training ran in Docker on the GPU desktop's NVIDIA A10G. The successful
training container had networking disabled, a read-only dataset mount,
eight CPU cores and a 16 GB memory limit. Feature extraction for 1,592
images took 32.1 seconds; the successful training and two exports took
180.4 seconds. Timing is diagnostic, not the model-selection criterion.

The frozen experiment learns a linear material head. The fine-tuned
experiment also updates the last two vision blocks, final normalization
and attention-pooling head. It changed 45 encoder tensors.

Validation uses 200 MINC annotations and 12 camera material references.
The checkpoint objective is 70% public category accuracy and 30% camera
teacher agreement. The first epoch-four checkpoint won; epoch six tied.
Test predictions were evaluated after that selection.

| Model | MINC validation | Camera validation agreement | Selection objective |
|---|---:|---:|---:|
| Frozen encoder + learned head | 154/200 (77.0%) | 6/12 (50.0%) | 0.689 |
| Fine-tuned encoder + learned head | 156/200 (78.0%) | 7/12 (58.3%) | 0.721 |

The accepted-label score threshold and temperature were selected from
validation. The frozen head uses temperature 1.0 and score ≥0.825; the
fine-tuned model uses temperature 2.0 and score ≥0.775. Both require a
score margin ≥0.05, a supported class and a non-rejection label.

Only **one camera material reference** was accepted during validation for
each variant. Thus the manifest's `qualified_for_acceptance` flag describes
the programmed validation rule; it does not establish reliable camera
precision from a large sample. Both variants also accepted two unresolved
validation crops, whose correctness cannot be scored.

## Held-out test results

The public test has 200 independently annotated images. The camera test
has 29 crops: 16 weak target-material references, 12 skin/food rejection
references and one disagreement. These are different reference types and
are reported separately.

| Model | MINC category accuracy | Camera agreement with Nova |
|---|---:|---:|
| Unchanged zero-shot base/384 | 85/200 (42.5%) | 2/16 (12.5%) |
| Frozen encoder + learned head | 150/200 (75.0%) | 5/16 (31.3%) |
| Fine-tuned encoder + learned head | **153/200 (76.5%)** | **8/16 (50.0%)** |

This comparison uses identical held-out pixels and the same 15-label
vocabulary. It is not a comparison to the earlier 200-image pilot or
20-reference camera qualification.

| Model | Public accepted | Public correct among accepted | Camera accepted | Camera target references correct among accepted |
|---|---:|---:|---:|---:|
| Zero-shot | 0/200 | — | 0/29 | — |
| Frozen encoder + learned head | 98/200 (49.0%) | 86/98 (87.8%) | 3/29 (10.3%) | 2/2 |
| Fine-tuned encoder + learned head | 92/200 (46.0%) | 86/92 (93.5%) | 1/29 (3.4%) | 1/1 |

The frozen head's third camera acceptance is a false material identification
on a rejection reference. The fine-tuned model accepts none of the 12
skin/food rejection references. Neither accepts the unresolved test crop.
One accepted camera material is too small a sample for a precision claim.

The fine-tuned model improves public top-choice accuracy by 34 percentage
points and camera teacher agreement by 37.5 points over zero-shot.
Fine-tuning adds 1.5 public points and three correct camera references over
the frozen learned head. The combined MINC/camera training run does not
isolate the contribution of VLM labels from that of the annotated control;
a public-only training ablation would be needed for that claim.

Camera top-choice agreement by class:

| Weak teacher label | Zero-shot correct | Frozen head correct | Fine-tuned correct |
|---|---:|---:|---:|
| Fabric | 2/4 | 1/4 | 1/4 |
| Glass | 0/1 | 0/1 | 0/1 |
| Paper | 0/6 | 4/6 | 6/6 |
| Plastic | 0/1 | 0/1 | 0/1 |
| Rubber | 0/2 | 0/2 | 0/2 |
| Wood | 0/2 | 0/2 | 1/2 |

The improvement is concentrated in paper and wood. Fabric regresses on this
small test. These results do not support a claim that every camera material
category improved.

## Export and NUC verification

Both exports retain the FP32 normalized vision encoder, original text
embeddings, learned head, exact preprocessing and checksums. Each passes
a source-versus-OpenVINO embedding comparison on six real images plus
gray, red and noise inputs, with a maximum absolute-error gate of 1e-4.

The complete test split was replayed through the application's
`MaterialClassifier` on the NUC CPU with two inference threads. Both variants
matched all 229 expected top candidates and acceptance decisions, totaling
458 checked predictions.

| Variant | Source/export embedding maximum error | GPU/NUC score or margin maximum error | NUC median / p95 per crop |
|---|---:|---:|---:|
| Frozen head | 6.56e-7 | 7.97e-6 | 481.7 / 505.3 ms |
| Fine-tuned | 1.69e-6 | 8.49e-6 | 481.7 / 500.0 ms |

Both numerical checks remain below the 1e-4 gate. The replay reproduces the
reported public category counts, camera teacher agreement, rejection and
abstention. The complete receipt is recorded alongside `results.json` as
`nuc-verification.json`. These timings describe sequential crop inference,
not concurrent camera-service throughput.

## Artifacts and reproduction

Code, overall plan and commands are versioned in this repository. Private
camera pixels, teacher responses, feature caches and model weights remain
outside Git under:

```text
~/.local/state/bolt-rgb/material-distillation-2026-10-28/
  dataset/
    dataset.json
    controls.json
    teacher-manifest.json
    teacher-requests/
    teacher-labels.json
  output/
    feature-manifest.json
    evaluation-rows.json
    evaluation-logits.npz
    results.json
    zero-shot-evaluation.json
    nuc-verification.json
    receipts/
    artifacts/{frozen-head,fine-tuned}/
```

The successful training image ID is
`sha256:42cac1973d9af17c05781662abbf309e86d70420a31a985396a4b678898b0443`.
Container inspection, logs and installed package inventory are preserved.
Private source snapshots/checksums preserve the code used for the receipts;
the published extractor additionally protects later-session exact duplicates.
The first feature attempt lacked tokenizer dependencies; they were pinned
before the successful extraction. The first training attempt was stopped
when optional processor-file network probes were discovered; the corrected
job loaded the cached pinned revision by local path in offline mode.

The validation-selected artifact is installed at:

```text
~/.cache/bolt-rgb/models/siglip2-material-student-384-20261028
```

Select it when starting the local service:

```bash
rgb-console -- --materials-model-dir \
  "$HOME/.cache/bolt-rgb/models/siglip2-material-student-384-20261028"
```

An attaching console uses the already-running service's model configuration.
The model was subsequently deployed as the console's persistent default through
the `siglip2-material-student-384` cache symlink. The owning Qt console was
restarted, and live Femto Bolt crop readings matched the fine-tuned manifest
and learned-score policy. Capture remained paused. Deployment receipts are
under the private run directory's `deployment/` folder.

The application supports both original and learned-head artifacts, retains
crop evidence and head/data provenance, and displays the appropriate score
type in Live and Records. Arrow/Parquet JSON fields preserve the full learned
score and calibration policy without converting it to VLM confidence.

Validation completed with 276 application tests passing, three trainer tests
passing in the model environment, and 31 JavaScript tests passing. The
application-only environment skips the separate PyTorch trainer test module.
Compilation and whitespace checks also pass.

## Next experiment

Collect new sessions with deliberate, varied examples of plastic, glass,
paper, fabric, wood, metal, rubber and cardboard. Reserve physically
different objects for the next evaluation set and review uncertain or
disagreeing teacher labels before adding hard targets. Current training
contains no camera paper examples and is dominated by skin rejection crops.

Compare public-only and public-plus-camera training on the new held-out
objects. Add reviewed rejection examples, including food and unsupported
surfaces. Evaluate whether segmentation masks or multiple material labels
reduce background contamination. Keep this test set fixed and report any
future reuse explicitly.
