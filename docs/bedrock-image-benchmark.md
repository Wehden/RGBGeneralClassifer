# Bedrock camera-image preflight and lifecycle

Checked October 26, 2026, through AWS profile `default`, source region
`us-west-2`. This is a preflight comparison; the app is not implemented yet.

## Image workload

Captured a complete color/depth/IR frame set from the attached Femto Bolt
using Orbbec SDK 2.1.2. Acquired color at 1280 × 720 and depth/IR at 640 × 576,
with 15 FPS stream profiles. Registered depth to color through the SDK.

Used one shared scene across all calls. Encoded JPEGs at width 640 and quality
75; color and registered depth were 640 × 360, and native IR was 640 × 576.

The prompt requested up to three visible objects with a concise label,
confidence, and visible materials, returned as JSON. The three-image workload
also explained the depth legend and that IR remains in native coordinates.
Both workloads used `maxTokens=240`, `temperature=0`, no forced tools, and
the default Standard service tier. Nova 2 Lite used
`reasoningConfig={"type":"disabled"}`.

Calls were sequential and models were interleaved. Each model received five
color-image calls and three calls with color, depth preview, and IR preview.

| Model | Color median | Color range | Three-image median | Three-image range | Valid JSON |
| --- | --- | --- | --- | --- | --- |
| Nova Lite v1 | 0.771 s | 0.618–1.059 s | 1.015 s | 0.986–1.104 s | 8/8 |
| Nova 2 Lite | 0.882 s | 0.696–1.430 s | 1.346 s | 0.817–1.881 s | 8/8 |
| Claude Haiku 4.5 | 1.236 s | 1.177–1.493 s | 1.410 s | 1.379–1.925 s | 8/8 |

The timers include the complete client Converse call. They exclude acquisition
and image encoding. Mean output token counts for color were 57.4, 106.0, and
132.0 respectively; models tokenize and format their answers differently.

Validation checked object-array bounds, labels, numeric confidence in [0,1],
and material string arrays. It did not score object or material correctness.
Some responses contained JSON Markdown fences, which the parser removed.

These samples do not establish sustained one-Hz operation or a latency SLA.
The existing app's fuller per-crop schema, more objects, and larger images
need their own checks. Treat one Hz as an observed reference point; make
cadence configurable and avoid accumulating stale frames.

Earlier Nova Lite three-image requests that forced a tool invocation failed
twice with `ModelErrorException`. The successful ordinary-JSON workload shows
that this failed request style should not determine the vision-model choice.
Nova's model cards currently list provider-enforced Structured Outputs as
unsupported, so application validation remains necessary.

Private imagery, prompts, request IDs, outputs, and measured timings are saved
outside the repository:

```text
/home/ANT.AMAZON.COM/kwehden/.local/state/bolt-rgb/model-benchmarks/20261026T164833Z/
```

## Lifecycle

Live `GetFoundationModel` queries in `us-west-2` returned `ACTIVE` for all
three models, with no Legacy or EOL timestamps in their lifecycle objects.
The original responses are saved as `lifecycle.json` beside the benchmark.

AWS model cards report:

| Model | Current state | EOL no sooner than | Announced EOL | Legacy period |
| --- | --- | --- | --- | --- |
| Claude Haiku 4.5 | Active | October 16, 2026 | N/A | At least six months |
| Nova Lite v1 | Active | December 5, 2025 | No scheduled date found | At least six months |
| Nova 2 Lite | Active | December 2, 2026 | N/A | At least six months |

Haiku 4.5's minimum-availability date has passed. That does not mean the model
is deprecated: AWS still marks it Active and reports EOL as N/A. The applicable
model card lists at least six months in Legacy before EOL. Keep the model
configurable and recheck lifecycle when selecting or updating the deployment.

Nova Lite's card also has an older "Model EOL date" field reading "No sooner
than 12/4/2025." This is a past lower bound, not a scheduled retirement date.
The card and live regional API both still report Active.

For Bedrock, AWS's own dates apply; another provider's direct-API dates should
not be substituted. Model lifecycle can differ by region.

## Updated proposal

Select `us.amazon.nova-2-lite-v1:0` for default live VLM analysis, with reasoning
explicitly disabled, compact JSON, and local validation. The choice favors the
newer model while accepting the measured median increase of 0.111 seconds for
color and 0.331 seconds for the three-image workload. A label-quality comparison
remains necessary.

Keep Nova Lite v1 and Haiku 4.5 configurable for comparison or deliberate model
changes. Do not silently switch providers or models after an error; retain the
attempt history and make the selected model visible. Cadence remains adjustable;
one Hz is an observed reference point, not a limit or requirement.

## Primary references

- Haiku 4.5 model card:
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html
- Nova Lite model card:
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html
- Nova 2 Lite model card:
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html
- Lifecycle policy for models launched before September 7, 2026:
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html
- Nova 2 Converse and reasoning configuration:
  https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-converse-api.html
