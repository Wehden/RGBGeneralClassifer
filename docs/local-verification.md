# Local implementation verification — 2026-10-26

The implementation is available for local use. This ledger distinguishes
contract tests and actual hardware/provider checks from remaining acceptance
qualification. It does not claim that every procedure in the design was
completed, or that model confidence establishes recognition accuracy.

Reference environment: Ubuntu 24.04, Python 3.12.3, base repository commit
`5ade37d` plus the implementation worktree, attached Femto Bolt firmware 1.1.3.
The host has a 13th Gen Intel Core i7-1360P with 16 logical CPUs.
Local dependencies are pinned in `requirements-local.txt`; Qt dependencies
are in `requirements-desktop.txt`. CPU inference was used; no discrete GPU
was required. Bedrock used the `default` profile in `us-west-2`, without changing
AWS credential files. Images, raw responses and validation archives are kept
outside the repository in `~/.local/state/bolt-rgb/validation`.

The baseline automated run passed **128 Python tests** and **seven browser
behavior tests**. The desktop tests are independent of Qt, so ordinary
headless CI retains the same test command. Actual Qt/X11 checks are recorded
separately below.

## Capture, API and concurrency

| Check | Result |
| --- | --- |
| Actual Bolt capture | 1,180.385 seconds; 17,654 accepted synchronized frames; 14.956 FPS delivered; zero incomplete frames, skew rejects or reconnects in that run. |
| Actual profiles | Color 1280×720 MJPG; native depth/IR 640×576 Y16, configured at 15 FPS; SDK registered depth 1280×720. |
| Exact sensor round trips | Native depth and IR PNGs match the NPZ arrays exactly as uint16; retained original color JPEG decodes to the captured color array; registered preview matches color dimensions. |
| Registration | Actual edge overlay visually inspected; calibration/profiles retained. This is not metrology certification or a completed synthetic reprojection oracle. |
| Eight viewers, 80 seconds | Health p50 1.982 ms, p95 3.369 ms; all three channels available; maximum 8 cached frames and 49,949,965 bytes. |
| Eight viewers plus 5-second provider stub | 300.367-second measurement, 299 status samples; health p50 2.015 ms, p95 4.225 ms; all channels available; maximum 8 cached frames and 50,236,497 bytes. |
| Slow-provider queues | Observed cloud concurrency one and pending live capacity one; replacement counters rose. The provider was a timing stub, not a recognition-quality test. |
| Ownership and restart | Serial-based SDK ownership and passive discovery reject a second runner before acquisition; V4L2 explicit serial shares the lock. Actual SIGTERM shutdown took 2.357 seconds; immediate restart and saved fallback/pause restoration passed. |

The slow-provider process remained running for other checks after its
five-minute measurement; later replacement totals are not five-minute counts.
Cache measurements remained below the configured 8-frame/128-MiB limits.
Freshness, skew rejection, missing channels, errors, raw-array immutability
and the stale-detector/live-stream behavior also have synthetic contract tests.

## Actual full-schema Bedrock comparison

Ten distinct Bolt frame IDs from one office view supplied 76 retained
detections. Twenty identical eligible batches were paired per model:
ordered image hashes, global box indices and prompt hashes match across
models. Each request included actual scene/crop JPEGs, registered depth
preview and native IR preview, with the full object/material schema.

| Model | Validated batches | Failed batches | Call p50 | Call p95 | Input/output tokens |
| --- | --- | --- | --- | --- | --- |
| Nova 2 Lite | 20/20 | 0 | 2,081 ms | 2,921 ms | 53,042 / 7,857 |
| Nova Lite | 16/20 | 4 | 2,091 ms | 3,109 ms | 111,482 / 4,082 |

These percentiles include all terminal Converse calls, including failed
responses, in a mixed workload of two to four crops per batch. They exclude
input preparation and archive time. They are separate from status-page
windows grouped by workload, which include successful requests only. The
comparison used a 30-second read timeout; the normal runner subsequently
validated real Nova 2 calls with its 15-second default. Nova 2 reasoning was
disabled.

Nova Lite's malformed or duplicate-key outputs remained explicit failures
and did not update labels. It remains an operator-selected fallback with a
documented reliability limitation. The measured comparison supports Nova 2
as the default; it does not establish a latency advantage for Nova Lite.

This dataset contains distinct captures of one view. Deliberately varied
sparse/multi-object scenes and a labeled, human-scored object/material
dataset remain unqualified. Protocol success is not recognition accuracy.
The original shorter image preflight is documented separately.

## UI and desktop

The retained browser behavior suite passed. Headless Chromium checks used
actual archived images with a refreshing fixture camera: all three views,
model save/dirty selection, pause/resume, modes, records and review drafts
worked without page errors or unsolicited location permission requests.
Hardware API reads were checked separately against the actual SDK.

The Qt console was also exercised on this desktop's actual Xwayland display
using `xcb`, with the actual camera service. Native toolbar switching loaded
color, depth and IR; Records and Dashboard navigation worked; native save
dialogs produced a readable sensor NPZ, uint16 IR PNG and archive ZIP. Its
compact layout and screenshots were inspected. The console uses the same
page/API controls as the browser. Lifecycle tests cover attachment, ownership,
close/start races, exited children, local URLs and incorrect listeners.

Actual closing checks confirmed that an attached service kept capturing
after its console closed, and a console-started service stopped and released
the listener when its window closed. All validation processes were stopped
after these checks. The desktop menu entry was installed and validated.

## Acceptance case ledger

| Case | Evidence and remaining qualification |
| --- | --- |
| V01 | Actual singleton ownership, default loopback, shutdown and immediate restart passed; alias and ownership contracts tested. |
| V02 | Actual 19.7-minute three-channel capture passed with measured rates/counters above. |
| V03 | Actual registered dimensions, calibration and visual edges checked; calibrated synthetic reprojection oracle remains pending. |
| V04 | Known-array scale/statistics/null/immutability tests and exact real raw PNG round trips passed. |
| V05 | Pinned IDs, NPZ, JPEG/PNG, cache errors, viewer slots and stable archive capture tested; actual channel/API reads passed. |
| V06 | Missing/skew/stale fixtures and normal restart tested; physical USB unplug/replug and an ordinary V4L2 camera remain pending. |
| V07 | Transaction/recovery faults, job links and saved settings tested; actual orderly restart passed. Killing a paid in-flight RPC and all physical filesystem fault boundaries remain pending. |
| V08 | Stable IDs, conflicting replay, history, mode/threshold revisions and empty observations tested. |
| V09 | Actual JPEG bytes in stubbed requests; complete validator, duplicate keys, nonfinite values, refusal/truncation and invalid schemas tested. Actual Nova 2 full-schema responses passed. |
| V10 | Timeout/error/backoff behavior tested and normal profile calls passed. Actual denied/expired credentials and rotation were not deliberately induced. |
| V11 | Actual camera, eight viewers and five-second provider stub passed the five-minute measurement. |
| V12 | Pinned manual policy, pause, fairness, capacity and live replacement tested; browser controls passed. |
| V13 | Model policy changes/pinning and explicit fallback/restore tested; saved fallback/pause verified on actual restart. Nova Lite's measured 80% schema success is below the proposed 90% functional acceptance target. |
| V14 | Asset/write/commit fault injection prevents false acknowledgments and unsaved cloud requests; physical whole-disk exhaustion was not induced. |
| V15 | Typed exports, SQLite snapshot consistency, evidence/hash joins and admission tested; large-volume qualification remains pending. |
| V16 | Primary protocol validation passed 20/20 paired batches and subsequent real CLI calls; scene diversity and human quality scoring remain pending. |
| V17 | Optional GPU/segmentation backend not shipped or required for this baseline. |
| V18 | Existing Redis/board/browser contracts retained, with new SQLite tests. |
| V19 | Continuous track loss, age/session/policy/manual guards tested; basic IoU association remains a limitation, without a physical object-replacement quality claim. |
| V20 | Chromium fixture UI and actual Qt/hardware channel checks passed; no unsolicited location prompt. |

## Reproduce

### Revision 2: capture rate, capture pause and 3D

The updated automated suite passed **143 Python tests** and **14 browser
behavior tests**. New cases cover a fast provider sharing the automatic
capture interval, pause acknowledgments versus live dispatch admission,
intentional snapshots while paused, saved controls, actual metric/RGB
point layout, distortion, invalid geometry/HTTP responses, and canceled
late browser responses.

Fourteen desktop checks passed with the actual Bolt on X11: RGB/depth/IR
and 3D views, native orbit/zoom, capture pause, changing live frame IDs
without new observations while paused, an edited interval surviving
polling, saved interval, manual one-shot capture, resume, the one-minute
limit, point-cloud download and return to RGB. After resume, the record
count stayed unchanged for the first 58 seconds and increased by exactly
one after 63 seconds. These checks used a private camera archive with VLM
disabled; provider cadence is separately covered by the fast-provider test.

The real WebGL viewer rendered approximately 10,225 valid colored points
at up to 4 Hz, with a non-background pixel readback. A separate forced
CPU-canvas browser check rendered 10,209 points with 9,492 colored pixels,
switched to depth shading, canceled polling on exit, and reported zero
page errors. This does not certify depth accuracy or fused reconstruction.

Private artifacts are in `validation/capture-cloud`. The normal console
was reopened with a saved 60-second interval, the previous VLM pause state
restored, the original archive retained, and profile `default`/Nova 2 Lite.
A post-update real Nova 2 request validated successfully; direct point-cloud
read/decode also passed. The earlier slow-provider soak above describes the
baseline scheduler, rather than a repeat of that load under the new interval.

### Commands

```sh
.venv/bin/python -m pytest -q
node --test tests/js/*.test.cjs
.venv/bin/python -m compileall -q board rgb common scripts
.venv/bin/python -m pip check

# Read-only camera/API soak, against an already running service:
.venv/bin/python scripts/verify-local.py --seconds 900 --viewers 8 \
  --output "$HOME/.local/state/bolt-rgb/capture-check.json"

# Explicit paid provider comparison; use a new private archive:
.venv/bin/python scripts/benchmark-bedrock-local.py --invoke-bedrock \
  --state-dir "$HOME/.local/state/bolt-rgb/model-check"
```

The actual X11 smoke artifacts and private screenshots are under
`validation/desktop`; provider comparison summaries are under
`validation/full-pipeline`. No private camera imagery is committed.
