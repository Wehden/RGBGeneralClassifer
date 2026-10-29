# Fine-tuning the local object and material models

2026-10-28. Both model families are adaptable. The training path uses the
source PyTorch checkpoints; the current ONNX and OpenVINO artifacts are
inference exports.

## What we can train

| Component | Supported adaptation | Practical first experiment |
|---|---|---|
| YOLOE26 object detection and segmentation | Fine-tune a matching text/visual checkpoint on labeled boxes or masks, with Ultralytics' class-aware fine-tuning trainer | Train an ordinary physical-object vocabulary while retaining segmentation for depth selection |
| SigLIP2 material classification | Train a classification head on frozen image features, or fine-tune the vision encoder and head | Frozen-feature classifier on human-checked material crops, followed by encoder tuning if the held-out result warrants it |

### YOLOE26

Our deployed `yoloe-26{s,m,l}-seg-pf.pt` source checkpoints are prompt-free.
Ultralytics explicitly says these are not directly trainable and do not
support `set_classes()`. We would start from the matching
`yoloe-26{s,m,l}-seg.pt` checkpoint.

For segmentation fine-tuning, the official trainer is `YOLOEPESegTrainer`.
For detection-only fine-tuning, use the corresponding model YAML, load the
segmentation weights, and use `YOLOEPETrainer`. The official documentation
also describes linear probing with the backbone frozen.

The standard class-aware fine-tuning recipe fuses the dataset's class names
into the prediction head. The resulting model predicts those classes;
it does not preserve arbitrary new text prompts. We must decide whether
the deployment needs a fixed domain vocabulary or unrestricted exploration.
The original open-vocabulary detector can remain a separately selected
baseline.

The [box-agreement pilot](box-agreement-benchmark.md) found stronger aggregate
agreement for large, but most failures were label disagreements, and its
advantage did not hold in both frames. This supports investigating the
vocabulary and supervision rather than assuming size alone fixes the problem.
Segmentation supervision is useful for the planned native-depth selection
and external-volume work.

An inexpensive baseline before training is to configure a text-prompted
checkpoint with ordinary target classes such as person, chair, monitor,
bottle and backpack, then export that fixed vocabulary. This is vocabulary
selection, **not fine-tuning**. It should be compared with both prompt-free
and trained models on the same held-out scenes.

### SigLIP2

Hugging Face exposes `Siglip2ForImageClassification` with a trainable linear
classification head and labeled classification loss. Its configuration
supports single-label and multi-label classification. We can train on
material categories instead of using the current text-similarity ranking.

The fixed-resolution checkpoints actually used here load as Transformers'
`SiglipModel`, rather than the NaFlex `Siglip2Model`. The
[implemented training process](material-distillation-plan.md) uses
`AutoModel.vision_model` and the checkpoint's normalized pooled features,
with a separately trained head. This preserves our existing export contract.

The first experiment should cache the existing image encoder's features
and train a small material head. This changes the classifier while leaving
the encoder weights fixed. It is a frozen-feature or linear-probe baseline,
not full encoder fine-tuning. Our existing OpenVINO graph exposes the image
embedding, so this experiment can reuse the qualified encoder without
loading a text encoder during inference.

If that baseline fails on a held-out camera dataset, we can fine-tune the
vision encoder together with its classification head from the original
Hugging Face checkpoint. Start by updating later blocks before considering
the entire encoder. This is a proposed experiment, not a measured gain.

Mixed objects need an explicit target definition. A bottle may show metal,
plastic and rubber. For whole-object crops, a multi-label head can predict
visible constituent materials, with a separate dominant-material field
when supported. A single-label material head is better matched to annotated
surface patches that contain one dominant material. We must not train
either task against invented hidden composition.

The current cosine-similarity thresholds are not calibrated probabilities.
Changing prompts or thresholds is calibration/zero-shot adaptation, not
encoder fine-tuning. Any trained classifier needs validation-based acceptance
and rejection thresholds, evaluated at the same operating point across models.

## Where to train

The NUC13 has an i7-1360P, 32 GB RAM and CPU inference working for all three
tested SigLIP2 encoders. It can generate and cache features and train a small
head on them. This is the recommended first material experiment.

Use the GPU desktop for substantial YOLOE fine-tuning and SigLIP2 encoder
updates, then export the winning models back to the NUC. The desktop GPU
model and VRAM have not been checked, so batch size, resolution and the
largest feasible backbone remain unqualified. Full-backbone CPU training
is technically possible for some configurations but is a poor first use of
this machine.

After training, the existing installer/export workflow needs a trained-model
artifact input, class mapping and checksums. Validate preprocessing and
source/export output parity before evaluating the exported model on the NUC.
Frozen material heads can instead be packaged as small checked weight
artifacts alongside the existing encoder.

## Training data and evaluation

Two office frames and repeated views of the same objects are insufficient
to qualify a model trained for this deployment. Collect varied objects,
distances, orientations, lighting and backgrounds, including negative cases.

For YOLOE, annotate ordinary object labels and boxes; add instance masks
for segmentation training. For materials, annotate visible surface material,
mixed cases, uncertain appearance and out-of-scope examples. Record whether
the material is visually supported, verified from the physical object,
or only proposed by a model.

Nova can propose labels and prioritize disagreements for review. Its two
passes are weak supervision: the box pilot already contains stable
hallucinations. Do not promote agreement between Nova passes to verified
material composition or human ground truth.

Split by physical object and capture session before extracting crops.
Randomly splitting neighboring frames or crops of the same object would
leak appearance into the test set. Use a separate validation set for
threshold selection and keep the final test set untouched.

The public MINC dataset is useful for a material baseline, but it does not
replace camera-domain evaluation. Our original 200-image pilot did not
follow official train/test splits, so it must not become the held-out test
set for a model trained on overlapping MINC data.

Compare the unchanged zero-shot models, the frozen-feature material head,
and any fine-tuned model on identical held-out examples. Report object
precision/recall and box or mask overlap, material per-class precision/recall,
acceptance coverage, rejection errors, and performance on mixed/unknown
cases. Execution time is a feasibility measurement.

## Recommended order

1. Complete the paired material agreement test to expose current failure
   modes, then human-check a varied set of reference crops.
2. Train a frozen SigLIP2 material head on the NUC and compare it with
   base/224, base/384 and large/256 zero-shot baselines.
3. Compare a constrained object vocabulary with prompt-free YOLOE, then
   fine-tune the best object backbone on the GPU desktop if labels and masks
   support a meaningful held-out test.
4. Consider SigLIP2 encoder updates if the frozen head cannot meet the
   material target on held-out camera data.

This document verifies supported training paths and proposes experiments.
The two YOLOE fine-tuning trainers and the SigLIP2 classification model were
imported successfully from the installed training environment; a 12-label
multi-label SigLIP2 configuration was also checked.
No encoder or detector has been fine-tuned in this investigation.

## Primary sources checked

- [Ultralytics YOLOE: checkpoint modes, fine-tuning trainers and linear probing](https://docs.ultralytics.com/models/yoloe/)
- [Hugging Face SigLIP2 API: classification model and configuration](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [Google SigLIP2 base/224 model card](https://huggingface.co/google/siglip2-base-patch16-224)
- Installed training implementation:
  `transformers==4.57.6`, `models/siglip2/modeling_siglip2.py`;
  `ultralytics==8.4.188`.
