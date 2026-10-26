# Portable networking and classification acceptance, 2026-10-26 UTC

Application source: `7fce583` on PR #3. This is a manual deployment; neither
PR #3 nor the supporting LaminarAuth PR #37 was merged by this work.

## Installed artifacts

| Component | Installed immutable identity |
| --- | --- |
| RGB web and ingest | `localhost:30500/rgb-classifier@sha256:34330f451a905de306c1f27f2720b5910d5d4ae119c3a33c5a95eee973a7cbef` |
| Uno Q native image | `rgb-classifier:unoq-portable-materials-20261026-v3`, image ID `sha256:6a3a68d7566a042f32fb4d5e437eb13dedfc23715fd94db698e7c1ea984c3421` |
| Materials graph | `23960eb842490d643721b1bd14c5c8491798343d91b3320e6511a4d5388a6bcf` |
| Objects graph | `8120cf9b3f3cf1dfbbc768b2eb869b48a18f6958f7272e1ba59258c1030ae806` |
| LaminarAuth validator | `laminarflow:30500/laminarauth/validate@sha256:71c5f06ebaf37a10347cb4e8d717c32665ab081949e93eab0b53c81ddc74c9de` |

The separately approved authentication patch accepts Cloudflare service JWT
string audiences and signed issue times when `nbf` is absent. Signature,
issuer, expiration and current route grants remain checked. Its focused suite
passed 175 tests; its full suite passed 2,176 with 57 opt-in skips. PR #37 CI
passed. The existing dirty LaminarAuth checkout was preserved.

## Network and authorization checks

The real board resolves `rgb.wehden.com` through public DNS and uploads both
live frames and observations to its HTTPS origin. No board address is in the
service source mapping; ingest is ClusterIP-only. Redis and the existing 9B
Ollama pool remain internal and unchanged.

The dedicated machine credential received 200 for its source policy route,
403 for dashboard/history/another source, and 401 without the RGB source
bearer. Unsigned public access redirects to Access login. Owner Google
access was exercised in Chrome. Credentials and source images remain private.

A mode-0600 `RGB-Ethernet` DHCP/autoconnect profile was saved and loaded on the
Uno Q without switching its active Wi-Fi connection. No lab SSID/password was
supplied. Physical cable insertion and a move onto the lab network remain
unperformed acceptance steps. The profile helper and device protocol remove
the need to edit application IP addresses after that move.

## Capture and analysis checks

- Preview ran both local models while archive/image counts stayed at 2,880,
  the board outbox stayed empty, and no `rgb:live:*` Redis keys existed.
- Pausing local classification stopped inference (`inference_ms=null`) while
  the camera continued delivering frames. Starting resumed local inference.
- At detector threshold 0.45, the live materials graph took about 324–329 ms,
  returned zero accepted boxes and a highest proposal around 0.08. Recorded
  material observation `c0d36209-dad8-4ab6-aee8-feeb8d617687` retains all 100
  proposals and zero detections. The earlier six-run archived-frame benchmark
  is in [materials mode](materials-mode.md).
- Capture `4cd8e487-4411-47c7-895c-a73be05ed18d` remained `manual_only` after
  VLM resume. Its capture policy refers to paused settings revision 21.
- An individual UI request on held observation
  `cfbf8f1a-dd1e-47b3-a545-a23efac04736` completed only box 0. Analysis remains
  partial, 1/7 boxes, with the other six unread. Attempt
  `8a5f5d95-c6c9-43c3-a474-227c5d4d971a` returned screwdriver 0.95 versus the
  original pencil, plastic 0.98 and metal 0.95. Original detector data remains.
- Catch-up paging, fixed watermarks, deduplication, source scoping, capture-time
  pause and checkpoint/live scheduling are covered by automated tests. A
  whole-history catch-up was not invoked during production acceptance.
- An alias between accepted boxes and raw proposals initially made 18 buffered
  uploads fail validation. Accepted boxes now use copies. The buffered rows
  were backed up, repaired and acknowledged; no original evidence was deleted
  before acknowledgement. The outbox then drained to zero.

Arrow and Parquet downloads both passed real HTTP generation and local
PyArrow reads of all ten tables, schema 7. The snapshots each contained 2,901
observations and 15,213 detection rows; model/action streams retain their own
captured watermarks. Files were approximately 6.1 and 6.5 MB. No image data is
included in these table downloads.

The first verification process imported PyArrow and read downloads inside the
512 MiB web pod, exceeding its shared memory limit and causing one OOM restart.
Streaming the downloads to the workstation and reading them there succeeded.
That failure exposed a browser stream-end case: reconnection now changes the
resource URL, retries an ended zero-width image, and requires a decoded image
before showing LIVE. A real 640×480 camera image was observed in Chrome.

Runtime was restored to Objects, local classification running, recording on,
thresholds 0.45/0.65 and VLM unpaused, matching the pre-test operator values.
Temporary rollout pauses finish in-flight calls before restarting web.

Five valid PR findings were addressed: proposal dictionary aliasing, stale JPEG
serialization, pagination through empty scenes, per-endpoint checkpoint recovery,
and refreshing detail polling after individual requests. Regression coverage
includes 501 empty scenes and a two-endpoint job interrupted during its second
endpoint; resumption performs each successful batch once.

Validation: 94 Python tests and 11 browser behavior tests; JavaScript syntax,
model SHA checks, service build and PR CI. CI remains validation-only.

## Retained rollback

The previous prompt-free native image and pre-portable runtime/compose files
remain on the board. Original outbox backup:
`/home/arduino/rgb-classifier/data/outbox.before-proposal-copy-20261026.sqlite3`.
Pre-portable runtime and compose backups are under
`/home/arduino/rgb-classifier/`, with private runtime permissions retained.
The preceding service image is
`localhost:30500/rgb-classifier@sha256:7ec962c800ecc8d434172dfa48e73a37261c0e1fe4443ecc35d7f9f7697f3439`.
Rolling back the board to its old private-IP transport also requires restoring
the matching service path; do not pair it with the new source-only manifest.
