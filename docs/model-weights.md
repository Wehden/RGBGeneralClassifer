# Private material student releases

The weight registry is the private Hugging Face model repository
`kwehden/rgb-material-student`. GitHub holds code; the Hub holds exported
inference weights and their manifest. Training data and workspace stay local.
Downloading is an explicit deployment step. The running console keeps using
local files and does not contact the Hub for inference.

## Install the tooling

Run from this repository's checkout on ventunoq or Stepwise:

```sh
python3 -m venv .venv-artifacts
.venv-artifacts/bin/pip install -r requirements-artifacts.txt
```

This environment needs neither OpenVINO nor PyTorch. It does not replace the
console's `.venv`. A separate publisher token needs write access to the private
model repository; download machines use a read-only token. Tokens stay outside
Git. `--token-file` takes precedence over `HF_TOKEN` and normal Hub credentials.
Do not put a token value in command arguments or receipts.

## Publish an inference export

Publish from the machine holding the exported student directory. It must contain:

```text
manifest.json
vision.xml
vision.bin
text_embeddings.npy
classifier.npz
```

The original manifest must accompany the bytes. Do not regenerate checksums to
make an incomplete or changed copy pass validation.

```sh
.venv-artifacts/bin/python scripts/model-weights.py validate /path/to/student-export
.venv-artifacts/bin/python scripts/model-weights.py publish /path/to/student-export \
  --token-file "$HOME/Documents/HF.token"
```

Publish verifies every artifact against the manifest, copies an upload snapshot,
and verifies it again. One Hub commit contains the export, `release.json`, and
a model card using the manifest's license. Extra source files, such as camera
images or token files, are never included. The repo and selected branch must
already exist, and the repo must be private. A concurrent change to the selected
branch causes the optimistic commit to fail rather than silently replace another
release.

The command prints JSON containing `repo`, the full immutable `revision`, and
`manifest_sha256`. Keep that output as the release receipt. Publication does
not create a tag, change runtime settings, or restart any service.

Integrity checks do not establish model accuracy or acceptance qualification.
The tool preserves the manifest's qualification policy, including an unqualified
head. For exports calibrated before the accepted-rejection-reference fix, rerun
calibration/evaluation with the corrected trainer before treating the release
as qualified. This does not require retraining the encoder.

## Archive every completed weight experiment

Preserve completed learned weights on the private Hub alongside their original
configuration, checksums, upstream revisions, training provenance and
qualification policy. Keep experimental exports on named `experiments/` branches;
`publish` defaults to `main`, so always pass `--branch` when archiving a candidate.
Uploading an experiment does not select it for the console.

Create an archive branch from the current registry release once:

```sh
.venv-artifacts/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=(Path.home() / "Documents/HF.token").read_text().strip())
repo = "kwehden/rgb-material-student"
info = api.model_info(repo)
if not info.private:
    raise SystemExit("The weight registry must remain private")
api.create_branch(repo_id=repo, branch="experiments/my-material-run-fine-tuned",
                  revision=info.sha, exist_ok=True)
PY

.venv-artifacts/bin/python scripts/model-weights.py publish /path/to/student-export \
  --branch experiments/my-material-run-fine-tuned \
  --token-file "$HOME/Documents/HF.token"
```

The publisher uses that branch's head as the optimistic parent and sends the
branch explicitly to the Hub. Record the returned full commit SHA in the run's
publication receipt. Download that exact SHA, compare every inference file
against its original manifest, and retain the verification result. A named tag
can help discovery; deployments still use the immutable full SHA.

Head-screening runs save `head-state.pt` and `head-results.json` rather than
inference exports. Archive both on a research branch, verify the original
`classifier_sha256`, and include a checksum index. Do not pass these checkpoints
to the console loader or rewrite their metadata into a fabricated export.
Camera pixels, datasets, feature caches, teacher responses and credentials stay
outside the weight registry. Unchanged upstream encoders can be referenced by
their original Hub repository, full revision and source-file checksums.

The [November 2 sync receipt](model-weight-sync-20261102.md) lists the deployed
release, archived exports and checkpoint archive. Make publication and verified
download part of completion for future weight experiments.

## Fetch a pinned version on Stepwise

Use the full 40-character lowercase Hub commit SHA from the publish receipt.
Branch names, tags, and abbreviated hashes are rejected.

```sh
.venv-artifacts/bin/python scripts/model-weights.py fetch \
  --revision FULL_40_CHARACTER_COMMIT_SHA
```

The default repo is `kwehden/rgb-material-student`; `--repo` can select a different
private registry. The default cache is `~/.cache/bolt-rgb/models`. The command
downloads known filenames only, validates both the export manifest and release
checksums, and atomically installs a directory named
`rgb-material-student-<full-commit-sha>`. Its `origin.json` records the repo,
commit, and manifest digest. A previously cached version is checked again and
can be reused without network access.

To select that version for the next console service start:

```sh
.venv-artifacts/bin/python scripts/model-weights.py fetch \
  --revision FULL_40_CHARACTER_COMMIT_SHA --activate
```

Activation atomically replaces the `siglip2-material-student-384` symlink.
It refuses to overwrite a real model directory. It does not restart the
running console or change capture/VLM settings. Restart the owning camera
service when ready; attaching a new window alone does not reload its models.
Alternatively, use the returned directory with `--materials-model-dir`.

Rollback uses the same fetch command with the previous receipt's commit and
`--activate`, followed by an intentional service restart. Old versions remain
in the cache; neither fetch nor activation deletes them.

Before accepting a deployment, check `/api/vlm/status` reports the intended
local material model, manifest hash, and learned-head hash. The comparison API
must still pair its image and readings by exact frame and box. A Hub upload
alone does not prove successful local inference.
