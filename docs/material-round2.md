# Material model improvement: plan, evidence and repeatable process

Experiment date: 2026-11-02. The objective is better material recognition and
rejection on camera objects. Runtime is a feasibility check, not the ranking
criterion. This round does not change the console's selected artifact.

## Decisions and evidence

The original camera supervision is small: 28 material training crops, dominated
by fabric/plastic, versus 1,000 annotated MINC training crops and 86 camera
rejection crops after the original appearance purge. The legacy loss formula
allocates only 2.1494% of loss mass to camera materials, 5.2183% to camera
rejections and 92.6323% to MINC. A weighting experiment is therefore justified,
but it cannot create missing material evidence.

| Verifiable claim | Implementation / evidence |
| --- | --- |
| Camera material and rejection loss have explicit, separate shares | `loss_weights` in `training/materials/train.py`; effective masses and class masses saved in every report |
| Weights and class counts use training rows only | Weighting regression tests include arbitrary heldout labels/weights |
| Conflicting labels are excluded without inventing replacement composition | Eight preregistered original training exclusions; original labels and pixels stay immutable; four validation references are flagged only |
| Historical test outcomes cannot select this round's models | `--stage head` and `--validation-only`; full mode purges original appearance duplicates first, then removes test rows before optimizer/image extraction |
| Camera precision cannot borrow all its support from public references | `--minimum-camera-accepted-references 10`; all round-two candidates remain unqualified when this gate cannot be met |
| Teacher and student use the same pixels in the crop study | `--preserve-crop-jpeg`; exact 384×384 RGB JPEG bytes and hashes retained and verified |
| Paired crop views do not appear together in one teacher request | `--unique-groups-per-batch`; two blinded Nova 2 Lite passes, with the other pass and student predictions unavailable |
| Paired agreement comparisons have the same reference question | Crop evaluator pairs only resolved, identical materials within one detector group; changed/unresolved labels are counted separately |
| New sessions are not proof of new physical objects | Archive audit found 18 newer observations and 196 candidate crops, but no independently verified object identity/composition |
| Alternative encoders use the same selected dataset | DINOv2 probes retain original training/validation selection; no additional encoder-specific duplicate purge |

The original checkpoint objective remains 70% MINC validation top-choice
agreement plus 30% camera validation agreement. Validation has 200 MINC
references and only 12 camera material references: nine fabric and three
plastic, spanning eight approximate appearance groups. Camera labels are weak
VLM references. Repeated validation screening is exploratory; neither
agreement nor a higher checkpoint objective establishes field accuracy.

## Training results

Each head uses 401 AdamW steps with the original text-based initialization.
Encoder runs fine-tune the last two SigLIP2 vision blocks, final normalization,
pooling and material head for six epochs. Seed, inputs and selection rule are
fixed. The best validation checkpoint is retained.

| Cached-feature head | MINC correct / 200 | Camera correct / 12 | Objective |
| --- | ---: | ---: | ---: |
| Legacy replay | 154 | 6 | 0.6890 |
| Camera materials 20%, rejection 10%, MINC 70% | 150 | 6 | 0.6750 |
| Camera materials 40%, rejection 10%, MINC 50% | 146 | 6 | 0.6610 |
| Public-only control | 151 | 5 | 0.6535 |
| Eight reviewed exclusions + 20/10/70 weighting | 148 | 7 | 0.6930 |

Head weighting alone does not improve camera agreement. Removing conflicting
training targets gives one extra camera match, with a lower public score.
Do not compare a frozen head gain to the already fine-tuned deployed model
as though it were an end-to-end improvement.

| Encoder fine-tune | MINC correct / 200 | Camera correct / 12 | Objective |
| --- | ---: | ---: | ---: |
| Deployed historical baseline | 156 | 7 | 0.7210 |
| Camera materials 20%, rejection 10%, MINC 70% | 158 | 7 | 0.7280 |
| Camera materials 40%, rejection 10%, MINC 50% | 156 | 7 | 0.7210 |
| Public-only control | 158 | 6 | 0.7030 |
| Eight reviewed exclusions + 20/10/70 weighting | 157 | 7 | 0.7245 |

The best balanced candidate improves public validation by two examples
without improving camera agreement. The reviewed-label run gains one public
example and also leaves camera agreement unchanged. All four encoder runs
fail the stricter camera acceptance gate, and no replacement is deployed.
Their OpenVINO export checks pass the 1e-4 embedding parity gate.
The best balanced artifact was also copied to a private experiment directory
on the NUC and checked against every validation crop: all 229 top choices
match GPU evaluation, with maximum raw-logit absolute error 7.92e-5
(gate 1e-3). CPU results reproduce 158/200 MINC and 7/12 camera.
The active artifact alias still resolves to the original 2026-10-28 release.

## Other local classifiers

The original frozen SigLIP2 features were screened with 78 configurations of
ridge regression, class prototypes and weighted nearest-neighbor voting.
The best ridge scores 155/200 MINC and 7/12 camera, objective 0.7175, below the
deployed fine-tuned baseline. A camera-heavy ridge reaches 8/12 but falls to
147/200 MINC. Prototypes and neighbors perform worse.

An actual alternate encoder was also tested: pinned, Apache-2.0 DINOv2
ViT-S/14, approximately 88 MB of source weights. Both normalized CLS tokens
and mean patch tokens were extracted in an offline container, then screened
with the same 78 classifier configurations each.

| DINOv2 pooling, best validation-selected probe | MINC correct / 200 | Camera correct / 12 | Objective |
| --- | ---: | ---: | ---: |
| CLS token, ridge, camera 40%, lambda 0.01 | 119 | 6 | 0.5665 |
| Mean patch tokens, ridge, camera 40%, lambda 0.001 | 148 | 6 | 0.6680 |

Mean patch pooling is much better than CLS for this material dataset, but
neither beats SigLIP2. These are frozen small-encoder experiments, not a claim
that all DINO architectures or fine-tunes are inferior.
See [material-alternatives.md](material-alternatives.md) for source revisions,
licenses, MINC specialists, DINOv3 access constraints and further options.

## Paired crop study and teacher audit

Four synchronized RGB/depth/IR bundles were acquired from the attached camera
and retained privately. The crop pilot uses one RGB frame, eight eligible
YOLOE detector groups and four crop strategies each:

1. Stretch the detector box to a square.
2. Preserve aspect ratio with neutral padding.
3. Neutralize pixels outside the instance mask, then preserve aspect ratio.
4. Select the largest mask-interior square and enlarge it.

The largest square uses mask-interior distance, rather than assuming box
centre is foreground. Source bounds, native dimensions and mask fraction are
retained. Interior sides range from 27 to 167 source pixels; enlargement
cannot restore missing texture. Mask membership does not prove correct
object membership or a homogeneous material. Selection by detector confidence
and requiring all four variants excludes some difficult objects.

An initial batching pilot put all four views of each object in one request.
Those receipts remain private and are not used for the reported comparison.
The corrected run separates paired views into different teacher requests:
16 successful calls, two passes over 32 exact crop JPEGs. It yields 20 strong
agreements, 11 unresolved uncertain references and one disagreement.

| Strategy | Resolved weak references / 8 | Top-choice teacher matches | Top-three teacher matches |
| --- | ---: | ---: | ---: |
| Stretch | 6 | 4 | 5 |
| Aspect-preserving padding | 5 | 4 | 4 |
| Foreground mask | 5 | 2 | 4 |
| Interior patch | 4 | 2 | 2 |

These denominators differ, so the rates cannot establish a better crop method.
On matched same-material pairs, aspect-preserving padding gains one match
over stretch on four eligible groups; masking gains one and loses one on
four groups; interior patches tie on three groups. Only one group has the
same resolved material across all four variants. The deployed classifier
accepts one masked crop whose teacher reference is unresolved; that is not
an independently verified correct identification.

**Teacher failure:** both independent passes confidently call the whiteboard
“paper” for three variants, citing its writing as evidence despite the prompt
explicitly forbidding that inference. The mask also includes apparent ceiling
fragments near the whiteboard boundary. Repeating the same teacher gives
agreement on a shared semantic error. These labels must not become automatic
training targets or physical material ground truth. No replacement material
was invented.

The study is eight same-frame detector groups, not 32 independent objects.
All requests still receive context from the same scene and model. Physical
identity, novelty and composition remain unverified. These results are
descriptive weak-teacher agreement, not a fresh confirmatory test.

## Repeatable execution

Keep images, raw teacher responses, labels, feature caches and experimental
weights outside Git. Record dataset/label/control/source hashes, seed, source
revision, image ID, selected checkpoint and the actual validation objective.
Use new output directories; never overwrite an original run.

1. **Freeze and audit data.** Run the bound label review described in
   [material-label-review-and-holdout.md](material-label-review-and-holdout.md).
   Keep disputed validation references fixed. Register training exclusions
   before optimizing the reviewed ablation.
2. **Preload the container.** Use `training/materials/Dockerfile`, its pinned
   dependencies and the pinned SigLIP2 source cache. Copy the standalone
   trainer, private read-only data and cached features to the training host.
   AWS/HF credentials are unnecessary inside offline training containers.
3. **Run heads and encoder ablations.** Execute the helper on the training
   host with absolute directories:

   ```bash
   export MATERIAL_DATA_DIR=/private/original/data
   export MATERIAL_CACHE_DIR=/private/original/cache
   export MATERIAL_FEATURE_DIR=/private/original/output
   export MATERIAL_OUTPUT_DIR=/private/new-head-round
   export MATERIAL_REVIEW_MANIFEST=/private/review/manifest.json
   MATERIAL_STAGE=head MATERIAL_DEVICE=cpu bash training/materials/run-ablations.sh

   export MATERIAL_OUTPUT_DIR=/private/new-encoder-round
   MATERIAL_STAGE=train MATERIAL_DEVICE=gpu bash training/materials/run-ablations.sh
   ```

   Both stages use the strict camera acceptance gate. Full training uses
   `--validation-only`; default replay remains available separately. Container
   data/cache/source mounts are read-only; output is private, network disabled,
   resources bounded and no ports published. This round used an A10G.
   The desktop's installed driver libraries differed from its loaded driver.
   Matching extracted libraries were mounted only into the training processes,
   with explicit device mappings and `--runtime runc`. No driver installation,
   daemon configuration change or reboot occurred. The helper's optional
   `MATERIAL_DRIVER_DIR` supports an already-prepared isolated directory.
4. **Screen alternate heads/encoders.** Follow the pinned extraction and
   validation-only probe commands in `material-alternatives.md`. Retain only
   original `train`/`validation` rows; keep already excluded samples excluded.
   Freeze sample selection across encoders and report the complete candidate grid.
5. **Prepare and label exact crop evidence.** Use a new directory:

   ```bash
   python scripts/prepare-material-crop-study.py \
     --capture /private/frame.npz --detector /private/yoloe-26l-pf-640 \
     --output /private/crop-study
   python scripts/label-material-training.py \
     --dataset-dir /private/crop-study --preserve-crop-jpeg \
     --unique-groups-per-batch --invoke-bedrock --max-calls 16
   python scripts/evaluate-material-crop-study.py \
     --dataset-dir /private/crop-study --model-dir /private/deployed-model \
     --output /private/crop-study-evaluation.json
   ```

   Cloud labeling runs on the local credentialed machine using AWS `default`
   and Nova 2 Lite. Teacher receipts are resumable; failed/missing labels remain
   unresolved. A fixed call cap may leave a study incomplete and is not proof
   of complete labeling. Inspect exact bytes, masks and material evidence
   before allowing any weak reference into a new training dataset.
6. **Verify candidate artifacts.** Full training exports both variants and
   checks PyTorch/OpenVINO embedding parity on six actual crops plus three
   synthetic inputs, maximum absolute error ≤1e-4. Run a CPU check on the NUC
   as well. Do not update the console alias or HF default release when camera
   qualification or fresh-object evidence is insufficient.
7. **Confirm on known objects.** Build independently referenced, physically
   distinct development and confirmation sets, balanced across supported
   materials. All views/patches of an object share a split. Select and calibrate
   on development objects; evaluate a registered finalist once on confirmation
   objects. Report per-class/object metrics, rejection, accepted precision
   and coverage. More views of the same object do not create independent
   material references.

Private evidence is under
`~/.local/state/bolt-rgb/material-round2-2026-11-02/` on the local machine.
Source review and original probe receipts are in adjacent dated private
directories. Experimental containers and immutable outputs use a separate
dated directory on the training host. Raw images, internal hostnames,
credentials and model binaries are not part of the PR.

Verification: the combined application suite passes 376 tests with one
training-dependency skip; 27 trainer tests pass separately in the model
environment. The CPU candidate check above evaluates the complete validation
set. The independent image audit reconstructs all 32 stored JPEGs and source
bounds from the retained RGB/masks and verifies teacher request image hashes.

## Next improvement decision

The current evidence supports improving the dataset before scaling the model:
independently known surface materials, audited patch boundaries, class balance,
hard negatives and object-disjoint confirmation. Nova remains useful for
proposing labels and identifying uncertain examples, but its agreement must be
reviewed. Continue using the deployed artifact until a candidate demonstrates
camera improvement and qualifies on sufficient independent references.
