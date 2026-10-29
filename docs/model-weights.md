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

## Publish from ventunoq

Once the NUC's exported student directory reaches ventunoq, it must contain:

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
  --token-file "$HOME/Documents/HFToken.key"
```

Publish verifies every artifact against the manifest, copies an upload snapshot,
and verifies it again. One Hub commit contains the export, `release.json`, and
a model card using the manifest's license. Extra source files, such as camera
images or token files, are never included. The repo must already exist and be
private. A concurrent change to its main branch causes the optimistic commit
to fail rather than silently replace another release.

The command prints JSON containing `repo`, the full immutable `revision`, and
`manifest_sha256`. Keep that output as the release receipt. Publication does
not create a tag, change runtime settings, or restart any service.

Integrity checks do not establish model accuracy or acceptance qualification.
The tool preserves the manifest's qualification policy, including an unqualified
head. For exports calibrated before the accepted-rejection-reference fix, rerun
calibration/evaluation with the corrected trainer before treating the release
as qualified. This does not require retraining the encoder.

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
