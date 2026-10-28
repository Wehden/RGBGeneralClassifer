# Repeatable process: VLM material labels to a local student

Follow the [overall plan](material-distillation-plan.md). Use a new private
run directory for each dataset snapshot and preserve the source archive.
The first run's outcomes are in [the results](material-distillation-results.md).

## 1. Export a frozen capture snapshot

Use the retained RGB observation frames. The extractor verifies asset hashes,
deduplicates exact pixels, groups similar crops and reserves later-session
groups for evaluation.

```bash
run_dir="$HOME/.local/state/bolt-rgb/material-distillation-2026-10-28"
.venv/bin/python scripts/prepare-material-training.py \
  --archive "$HOME/.local/share/bolt-rgb" \
  --output "$run_dir/dataset" --max-crops 192
```

The script rejects an existing `dataset.json`; use a new run directory for
a new snapshot. Inspect `split_counts`, session assignments and the original
crop/scene evidence before labeling. With fewer than three sessions, there
is no full session-held-out train/validation/test qualification.

## 2. Create the VLM labels

The preparation command is offline. It validates hashes and builds numbered
scene/crop JPEG evidence without exposing detector labels:

```bash
.venv/bin/python scripts/label-material-training.py \
  --dataset-dir "$run_dir/dataset"
```

Run the two Nova 2 Lite passes through the configured `default` AWS profile:

```bash
.venv/bin/python scripts/label-material-training.py \
  --dataset-dir "$run_dir/dataset" --invoke-bedrock --max-calls 128
```

The 192-crop snapshot requires 96 calls, with at most four crops per request.
Resume using the same command. Recorded attempts, including failed or
interrupted calls, are never replayed. To perform a new independent label
experiment, use a separate copied dataset/run directory.

Review `teacher-labels.json`: consensus by split and class, uncertainty,
mixed materials, disagreements, missing rows and formatting recovery.
These labels remain weak VLM supervision. Preserve every raw response and
the parser source hash for auditing and future human correction.

## 3. Add disjoint labeled controls

The official MINC archive and original pilot manifest are prerequisites.
The split file contains the archive's `labels/train1.txt`, `validate1.txt`
and `test1.txt` verbatim, extracted without unpacking arbitrary paths.
An example extraction:

```bash
.venv/bin/python - "$run_dir" <<'PY'
import json, sys, tarfile
from pathlib import Path
run_dir = Path(sys.argv[1])
archive = Path.home()/'.cache/bolt-rgb/benchmarks/minc-2500.tar.gz'
splits = {}
with tarfile.open(archive, 'r|gz') as members:
    for item in members:
        if item.isfile() and any(item.name.endswith('/labels/'+name)
                for name in ['train1.txt', 'validate1.txt', 'test1.txt']):
            splits[item.name] = members.extractfile(item).read().decode()
(run_dir/'minc-official-splits.json').write_text(json.dumps(splits))
PY

.venv/bin/python scripts/prepare-material-controls.py \
  --archive "$HOME/.cache/bolt-rgb/benchmarks/minc-2500.tar.gz" \
  --splits "$run_dir/minc-official-splits.json" \
  --prior-dataset "$HOME/.local/state/bolt-rgb/qualification-2026-10-28/minc-dataset.json" \
  --output "$run_dir/dataset"
```

This selects official split 1 and excludes the original 200 pilot images.
The material categories and deterministic selection are in `controls.json`.
Do not tune against the selected test examples.

## 4. Build the GPU container and transfer inputs

Set the desktop hostname using your existing SSH configuration:

```bash
desktop_host="${GPU_TRAINING_HOST:?Set GPU_TRAINING_HOST to your GPU desktop SSH host}"
ssh "$desktop_host" 'mkdir -p "$HOME/projects/bolt-rgb-material-training"/{container,data,output,cache}; chmod 700 "$HOME/projects/bolt-rgb-material-training"'
rsync -a --exclude=__pycache__ training/materials/ \
  "$desktop_host:projects/bolt-rgb-material-training/container/"
rsync -a --exclude=teacher-requests/ --exclude=teacher-evidence/ \
  --exclude=teacher-manifest.json "$run_dir/dataset/" \
  "$desktop_host:projects/bolt-rgb-material-training/data/"
ssh "$desktop_host" 'docker build -t bolt-rgb-material-training:2026-10-28 "$HOME/projects/bolt-rgb-material-training/container"'
```

The Dockerfile pins PyTorch/CUDA by image digest and the primary Python
dependencies by version. Record the built image ID and installed package
inventory alongside the run. Use separate desktop directories and image/job
names for a new experiment so the first run's outputs remain immutable.

No AWS credentials are needed on the desktop. The training dataset contains
the VLM label receipts' distilled annotation result, not credentials.

## 5. Cache source features and weights

Run the feature job with outbound network access to fetch the pinned public
checkpoint; it publishes no ports:

```bash
ssh "$desktop_host" '
training_root="$HOME/projects/bolt-rgb-material-training"
docker run -d --name bolt-material-features-20261028 \
  --gpus device=0 --user "$(id -u):$(id -g)" \
  --cpus 8 --memory 16g --shm-size 2g \
  --cap-drop ALL --security-opt no-new-privileges \
  --mount "type=bind,src=$training_root/data,dst=/data,readonly" \
  --mount "type=bind,src=$training_root/output,dst=/output" \
  --mount "type=bind,src=$training_root/cache,dst=/cache" \
  bolt-rgb-material-training:2026-10-28 \
  --stage features --data /data --output /output
'
ssh "$desktop_host" 'docker logs --tail 30 bolt-material-features-20261028; docker inspect --format "{{.State.Status}} {{.State.ExitCode}}" bolt-material-features-20261028'
```

Wait for exit code zero. `feature-manifest.json` records source model type,
revision, image hashes, feature checksum and preprocessing. The actual
fixed-resolution model uses Transformers' `Siglip` implementation.

## 6. Train, select and evaluate

After all teacher passes and the feature stage finish, use the cached weights
and features with networking disabled:

```bash
ssh "$desktop_host" '
training_root="$HOME/projects/bolt-rgb-material-training"
docker run -d --name bolt-material-train-20261028 \
  --gpus device=0 --user "$(id -u):$(id -g)" --network none \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --cpus 8 --memory 16g --shm-size 2g \
  --cap-drop ALL --security-opt no-new-privileges \
  --mount "type=bind,src=$training_root/data,dst=/data,readonly" \
  --mount "type=bind,src=$training_root/output,dst=/output" \
  --mount "type=bind,src=$training_root/cache,dst=/cache" \
  bolt-rgb-material-training:2026-10-28 \
  --stage train --data /data --output /output --epochs 6
'
ssh "$desktop_host" 'docker logs --tail 30 bolt-material-train-20261028; docker inspect --format "{{.State.Status}} {{.State.ExitCode}}" bolt-material-train-20261028'
```

Keep the frozen-head and fine-tuned experiments on identical splits.
Validation selects the checkpoint and calibrated acceptance policy. The
final held-out test reports both variants and the unchanged zero-shot
baseline. No mixed or uncertain camera example becomes a hard training
material merely to increase the labeled count.

Record `results.json`, evaluation rows/logits, training logs, container
inspection and both artifact manifests. A failed export-parity gate means
the artifact is not ready for local use.

## 7. Verify the artifact on the NUC

Copy the artifact directories and evaluation receipts back through SSH:

```bash
rsync -a "$desktop_host:projects/bolt-rgb-material-training/output/" \
  "$run_dir/output/"

.venv/bin/python scripts/verify-material-student.py \
  --data "$run_dir/dataset" --output "$run_dir/output"
```

Load the candidate through `rgb.materials.MaterialClassifier`, verify artifact
checksums, replay the same held-out crops and check class scores, abstention
and decisions against the container receipts. The replay saves
`nuc-verification.json`, including rejection and unresolved-case counts.
Its default replays the complete test split for both variants; use
`--splits validation test` to check both held-out splits. Run the repository
tests; the trainer suite needs the separate model environment with PyTorch,
Transformers and pytest:

```bash
.venv/bin/python -m pytest -q
.venv-models/bin/python -m pytest -q training/materials/test_train.py
node --test tests/js/app.test.cjs tests/js/records.test.cjs tests/js/pointcloud.test.cjs
```
Install the chosen artifact in a new model-cache directory and select it
using the local runner's `--materials-model-dir` option.

For the first run's validation-selected fine-tuned model:

```bash
student_dir="$HOME/.cache/bolt-rgb/models/siglip2-material-student-384-20261028"
test ! -e "$student_dir" && cp -a "$run_dir/output/artifacts/fine-tuned" "$student_dir"
rgb-console -- --materials-model-dir "$student_dir"
```

Runner options configure a newly started service. If the console attaches
to an existing service, restart that service with the material-model option
to select the new artifact. Use `--no-vlm` for local-only operation.

An accepted material needs a qualified threshold, enough training examples,
and a non-rejection class. The learned head has its own score and policy;
do not apply the old cosine-similarity threshold to its softmax output.

## 8. Decide what the results justify

Report public category accuracy separately from camera teacher agreement.
Include denominators, acceptance coverage, rejected/unreferenced examples,
class support and export parity. Record which classes are not supported
by enough training examples.

Use fresh capture sessions and reviewed labels for the next qualification
run. Preserve this dataset and test split; do not turn test failures into
training examples and keep calling the same test set held out.
