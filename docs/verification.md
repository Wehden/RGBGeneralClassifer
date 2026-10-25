# Verification record

Date: 2026-10-24, America/Los_Angeles. Initial deployment was built from the
`codex/trash-sorting-demo` work. Final PR and source commit are available in the
repository history. No merge is implied by deployment.

## Software and archive

- Initial service unit/integration tests: 13 passed. Includes input validation,
  configured-location authority, authentication, stable-UUID deduplication,
  pagination, image retrieval, separate opinions, ingest-role isolation, and
  Redis-outage startup recovery in the VLM worker.
- Service images built and pushed, then deployed by immutable registry digest.
- Real X8 camera observations reached Redis Streams with UUID, UTC timestamps,
  source ID, `Home desk` configured location, model hash, boxes, and images.
- Redis restart preserved the first stream ID `1792897683805-0`, existing
  observations, indexes, and referenced images. New ingestion resumed.
- Isolated RDB restore loaded 11 events, 11 UUID indexes, and 11 opinion entries
  from a completed snapshot. The test container had no network and no published
  port and was removed afterward. This was a restore check, not a scheduled
  or independently stored production backup.
- The web pod was moved from pragmatic to backdraft. The first event and its
  referenced JPEG remained readable through the shared NFS volume, then the
  temporary node constraint was removed. NFS remains a single-server dependency.
- A pod outside the allowed namespaces could not connect to Redis, web, or the
  ingest NodePort. The allowed board reached ingest health200, history404, and
  unsigned ingest401. No Redis/public/alternate history NodePort was added.

## Model and initial X8 prototype

- EfficientDet-Lite0 metadata v1 was downloaded from TensorFlow's official
  example URL. Kaggle's model-instance API returned Apache2.0 licensing.
- SHA-256 `2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b`.
- Fresh native ARM64 build on X8, not an imported application image. LiteRT
  input was uint8 `[1,320,320,3]`; outputs were `[1,25,4]`, `[1,25]`,
  `[1,25]`, `[1]`. A blank-frame inference returned no detections.
- Actual desk scene produced boxes on both keyboards and the monitor (`tv`
  is the COCO class). Observed unrestricted inference roughly168–275ms,
  about5.4FPS in a short viewer-load sample, 78–90MiB RAM and around260–277%
  CPU. These are short prototype measurements, not a sustained guarantee.
- Four concurrent streams returned200; fifth429. Health answered200 in12ms
  while all four viewers were active. Live boxes were visually inspected.
- The initial move interrupted the native build. Later the board restarted
  again, retained address192.168.1.231, and automatically resumed capture and
  drained its outbox. No reset cause was established.
- Thermal sensor reached87–91°C. Its passive threshold is75°C and critical
  threshold95°C. RGB was stopped and restart disabled. The user chose the Uno Q
  as primary and is adding X8 cooling separately. Temperature alone does not
  establish the cause of the restart. The old X8 bird/RGB images, containers,
  secrets, and outbox remain for rollback/testing.
- Board defaults now bound LiteRT to2 threads and capture to2FPS, with
  configurable limits. This was implemented after the unrestricted experiment.

## VLM

- Queried the live eisbox Ollama inventory: `qwen3.5:27b-q4_K_M` advertises
  vision capability. Real JPEGs were accepted by `/api/chat` and produced
  descriptions saved as separate opinions. Detection labels were unchanged.
- The initial startup worker failed if Redis was not ready; corrected and
  covered by an outage/retry regression test. Redis restart recovery was then
  exercised. Sampling skips/failures remain explicit. New responses carry
  output-limit metadata; older prototype responses predate this marker.
- No quorum accuracy, material truth, recyclability, or sorting-bin correctness
  is claimed. No trash-specific training or accuracy evaluation has occurred.

## Public authentication

- Protected root `rgb-root` registered to `rgb-classifier/rgb-web:80`.
- Initial grant is owner-only. `/onboarding` uses separate enrollment audience
  and the existing operator approval workflow, through LaminarAuth PR36.
- Public Cloudflare DNS/tunnel/access state read back after writes. Root returns
  an Access302; request form returns200. Direct unsigned origin returns403.
  All path checks include the live/history/image protected root.
- LAN router cached the hostname's earlier NXDOMAIN until about20:36 PDT.
  Public resolver and direct TLS edge checks already resolve/route correctly.
  Browser sign-in acceptance remains separate from these request checks.
- LaminarAuth implementation suite1253 passed/37 skipped; separate reviewer
  independently ran460 passed/19 skipped and reported no actionable findings.
  PR36 CI jobs passed. Source and infrastructure receipts are retained privately
  outside this repository; no credentials or validation emails were published.
- Live Google allowed-user, denied-user, wrong-valid-audience acceptance and
  request/approval/welcome round trip have not yet been exercised. Rollback
  instructions exist; an auth edge rollback rehearsal has not been performed.

## Remaining release checks

Uno Q model/camera/temperature checks, final service CI, and authenticated
browser acceptance are recorded below as they complete. Neither an access
redirect nor a Kubernetes Ready condition establishes those outcomes.
