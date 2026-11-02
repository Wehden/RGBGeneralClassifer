# Model weight synchronization — 2026-11-02

All persisted material student weights from the original and round-two runs are
preserved in the private Hugging Face repository `kwehden/rgb-material-student`.
The deployed student was already present; this sync added nine inference exports,
five head-screening checkpoints and nine kit runtime exports. Each added weight
file and its original configuration was downloaded by full Hub commit SHA and
compared against the source SHA-256. This verifies preservation, not accuracy.

The registry remains private; an unauthenticated request for its weight artifact
returned HTTP 401 before uploading. No camera pixels, training datasets, feature
caches, teacher responses or credentials were uploaded. Registry `main` remains
at the original deployed revision below, and the console's selected model alias
was unchanged.

## Material student inference exports

The original frozen-head baseline and all eight round-two exports have separate
`experiments/<tag>` branches. Named tags help discovery; use the full immutable
commit SHA with `scripts/model-weights.py fetch` for deployment.

| Export | Hub tag | Immutable Hub commit |
| --- | --- | --- |
| Deployed fine-tuned / 2026-10-28 | `siglip2-20261028-fine-tuned` | `f830a7fb92b17fc51c28dc4185c215c75018ad21` |
| 20261028 frozen head | `siglip2-20261028-frozen-head` | `e6b73a51dd21835722bc5397c8e6ffc628d6ade5` |
| 20261102 balanced20 frozen head | `siglip2-20261102-balanced20-frozen-head` | `453008df36e3a0a548c7f557d8947b2fb7a4317f` |
| 20261102 balanced20 fine tuned | `siglip2-20261102-balanced20-fine-tuned` | `ed963265aa889a8aa4f0da237d0d0501fcacb116` |
| 20261102 balanced40 frozen head | `siglip2-20261102-balanced40-frozen-head` | `b8f41a2e5dc978a7dc3d4b2b4a454a8355277b99` |
| 20261102 balanced40 fine tuned | `siglip2-20261102-balanced40-fine-tuned` | `1b8e3f5665d8da829830fa20f926757aaca36f1d` |
| 20261102 public only frozen head | `siglip2-20261102-public-only-frozen-head` | `1b5d77d99eb3e482eab1c8886031e6b4eb7af21f` |
| 20261102 public only fine tuned | `siglip2-20261102-public-only-fine-tuned` | `a5ebbea0f6c77d30d9ee431408a8a3e71e973b38` |
| 20261102 reviewed20 frozen head | `siglip2-20261102-reviewed20-frozen-head` | `318a5db57e8873e9223d96528612f49604d15100` |
| 20261102 reviewed20 fine tuned | `siglip2-20261102-reviewed20-fine-tuned` | `1a1563b1dca43ae37b5bff7689c90a39aa18e146` |

The original manifests and classifier policies are retained exactly. All eight
round-two exports have `qualified_for_acceptance: false`; archiving them does not
promote them. The original release's legacy qualification flag and its documented
sparse camera acceptance evidence are also retained without reinterpretation.

A real fetch of the balanced20 fine-tuned SHA through the application's download
command validated all inference files and installed a separate version directory
without activation. It remained unqualified for acceptance.

## Head-screening research checkpoints

The branch `experiments/siglip2-20261102-head-screening`, also tagged
`siglip2-20261102-head-screening`, contains five `head-state.pt` files
and their original `head-results.json` under `checkpoints/`: legacy, balanced20,
balanced40, public-only and reviewed20. Each checkpoint matches the original
training receipt's `classifier_sha256`. The archive's `checkpoint-index.json`
adds file checksums and pinned, unchanged DINOv2 encoder provenance.

Full archive commit: `91941f8b43ac01ddb1a442129ac4878eb9557ef2`.

These screening checkpoints are research artifacts and cannot be passed to the
console inference loader. DINOv2 encoders were frozen upstream checkpoints;
those probe experiments did not persist standalone fitted probe weights. Their
original model revisions, source-file checksums and 384 px preprocessing are
referenced in the index rather than presented as fine-tuned DINOv2 releases.

## Kit runtime inference exports

The branch `experiments/runtime-exports-20261102`, tagged `runtime-exports-20261102`,
contains exact inference graphs, original manifests and licenses under
`runtime-models/`. The `runtime-export-index.json` records file checksums.

Full archive commit: `6428a5014f364e1575094b9ffd5c86906f5eaac9`.

| Runtime export | Original artifact license |
| --- | --- |
| `yoloe-26n-pf-320` | AGPL-3.0 |
| `yoloe-26n-materials-320` | AGPL-3.0 |
| `yoloe-26l-pf-640` | AGPL-3.0 |
| `yoloe-26m-pf-640` | AGPL-3.0 |
| `yoloe-26s-pf-640` | AGPL-3.0 |
| `siglip2-base-224` | Apache-2.0 |
| `siglip2-base-384` | Apache-2.0 |
| `siglip2-large-256` | Apache-2.0 |
| `efficientdet-lite0` | Apache-2.0 |

These are preserved runtime exports, including the prompted YOLOE material graph;
this sync did not retrain them. Restore each directory using the full archive SHA
and verify its files against the index and original manifest. The student fetch
command handles student exports; it does not install these archive subdirectories.

## Receipts and repeatable completion

Detailed local receipts are retained outside Git:

```text
~/.local/state/bolt-rgb/hf-sync-2026-11-02/
  sync-receipt.json
  runtime-sync-receipt.json
  checkpoint-index.json
  fetch-verification/
```

Future completed weight runs follow the [publication process](model-weights.md#archive-every-completed-weight-experiment):
archive the original weights and configuration, record the full Hub SHA, verify a
download against the original checksums, and preserve the acceptance policy.
Publication and selection for deployment remain separate operations.
