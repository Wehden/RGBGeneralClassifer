# Local material alternatives: validation probes and research

2026-11-02. This compares alternative classifiers before adopting a larger
encoder. Recognition and rejection are the decision criteria. Execution
cost remains a feasibility measurement.

## Frozen-feature probes

Seventy-eight candidates reuse the original pinned SigLIP2 base/384 normalized
embeddings: class prototypes, weighted nearest-neighbor voting and
regularized linear regression, across six training-weight configurations.
Camera target materials and camera rejection samples are separate cohorts;
explicit cohort weights use class-balanced, teacher-weighted training samples.
Legacy weighting and a public-only control are included.

All fitting statistics come from training rows. Selection uses validation
only and the original objective of 70% annotated MINC top-choice accuracy
plus 30% camera weak-label agreement. No historical test predictions are
computed. This is exploratory validation screening across multiple
configurations; a fresh holdout is still needed for confirmation.

| Validation-selected candidate | MINC correct / 200 | Camera correct / 12 | Camera class-macro | Original objective |
|---|---:|---:|---:|---:|
| Existing fine-tuned model | 156 | 7 | 50.0% | **0.7210** |
| Frozen-feature ridge, legacy weights, lambda 0.0001 | 155 | 7 | 50.0% | 0.7175 |
| Frozen-feature ridge, camera-material 20%, rejection 10%, lambda 0.001 | 154 | 7 | 50.0% | 0.7140 |
| Frozen-feature ridge, camera-material 40%, rejection 20%, lambda 0.001 | 147 | 8 | 55.6% | 0.7145 |
| Public-only ridge, lambda 0.0001 | 155 | 6 | 33.3% | 0.6925 |
| Best class-prototype configuration | 125 | 5 | 38.9% | 0.5625 |
| Best weighted-neighbor configuration | 134 | 4 | 33.3% | 0.5690 |

These results do not establish a better replacement for the deployed model.
Extra camera weight can increase camera agreement at the expense of public
category accuracy. The best ridge head nearly matches the fine-tuned model
using its original frozen encoder, but does not beat the selection objective.
Prototypes and nearest neighbors perform worse.

Camera validation contains only nine fabric and three plastic references.
Those are weak VLM labels from a few office sessions, not independently
verified composition. A one-reference gain is too small and too narrow to
claim broader camera improvement. The probes report top-three, class-macro,
approximate-group macro and rejection-reference results separately.

Run:

```bash
OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-models/bin/python \
  scripts/probe-material-alternatives.py \
  --run-directory /private/material-training/output \
  --output /private/material-probes.json
```

The private initial receipt is
`~/.local/state/bolt-rgb/material-alternatives-2026-11-02/probe-results.json`.
It records all candidates, source hashes and the source-code hash.
Five regression tests cover cohort mass, training-only statistics,
separable reference classes and validation-only evaluation.
These heads are not exported or activated.

## Encoder and material-specific shortlist

| Technique | Why evaluate it | Current evidence / constraints |
|---|---|---|
| DINOv2 ViT-S/14, frozen CLS and mean-patch features | Self-supervised visual representations provide a different inductive bias; compare pooled object semantics with surface/patch evidence | Official small checkpoint is ungated, Apache-2.0 and about 88.25 MB FP32; no material result measured here yet |
| Surface-patch or foreground-patch SigLIP2 pooling | Reduce background and object-category influence while retaining the existing encoder | Requires teacher labels corresponding to the exact student pixels and a fresh object holdout |
| DINOv3 ViT-S/16 dense features | Newer dense-feature encoder; official small model has 21M parameters | Official HF checkpoint is manually gated and code/weights use the custom DINOv3 License; no download or acceptance of terms performed |
| Published MINC-specialist GoogLeNet/VGG/AlexNet models | Directly trained for 23 material categories and dense material recognition | Authors provide a 714 MB archive and report 85.2% mean-class MINC accuracy for GoogLeNet; Caffe integration, label mapping, model-license details and pretraining overlap need verification before a fair comparison |
| OpenCLIP checkpoints | Different contrastive pretraining is a possible alternative | Library code is MIT; each checkpoint's own license/provenance must be checked. No evidence yet that scaling another generic encoder improves this camera task |

The practical new-encoder candidate is DINOv2 small. Its weight storage fits
comfortably within the NUC's memory budget, but CPU inference behavior,
export parity, material utility and rejection must still be measured.
Desktop CPU containers can extract its features if the GPU is unavailable.
A visual encoder alone does not provide material labels: fit a supervised
head against reviewed targets, then calibrate on validation.

The separate DINOv2 extractor pins the source revision and downloads only
its configuration, processor configuration and safetensors weights:

```bash
.venv-models/bin/python scripts/cache-dinov2-material-features.py cache
```

Mount the returned snapshot directory and private dataset read-only in the
existing training container. Run extraction with network access disabled:

```bash
python scripts/cache-dinov2-material-features.py extract \
  --source-directory /models/ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --data /private/material-training/dataset \
  --rows /private/material-training/output/evaluation-rows.json \
  --output /private/dinov2-features --device cpu --batch-size 16
```

This validates sample hashes and writes separate normalized CLS and mean
patch features with a source/processor receipt. Extraction includes
immutable historical-test rows only as cached embeddings; it does not fit,
predict or select against those rows. The canonical processor resizes and
center-crops, which can discard parts of thin objects. Processor behavior
must be recorded when comparing encoders. This is an experiment extractor,
not a console-compatible exported material classifier.

The historical MINC control cannot automatically be treated as a clean
holdout for a published MINC-specialist model; training-image overlap may
exist. Published paper accuracy is not a result on our captures.
Depth can help isolate surfaces/foreground, but the Femto Bolt's depth/IR
channels do not by themselves identify physical composition.

## Primary sources

The following primary sources were checked on 2026-11-02. Copies and model
metadata are retained privately beside the probe receipt.

- DINOv2 authors' code and license:
  `https://github.com/facebookresearch/dinov2`
  `https://github.com/facebookresearch/dinov2/blob/main/LICENSE`
- Official DINOv2 small checkpoint:
  `https://huggingface.co/facebook/dinov2-small`
  pinned revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
- DINOv3 authors' code, model table and license:
  `https://github.com/facebookresearch/dinov3`
  `https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md`.
- Official DINOv3 small checkpoint metadata:
  `https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m`
  revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
- MINC authors' paper, models and annotation licensing:
  `https://opensurfaces.cs.cornell.edu/publications/minc/`.
- OpenCLIP official implementation and license:
  `https://github.com/mlfoundations/open_clip`
  `https://github.com/mlfoundations/open_clip/blob/main/LICENSE`.
- SigLIP2 authors' implementation:
  `https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md`.
