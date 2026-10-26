# Verification record

Current portable networking and capture acceptance: [2026-10-26 receipt](portable-acceptance-20261026.md). The dated sections below retain earlier deployment history.

Date: 2026-10-24, America/Los_Angeles. Initial deployment was built from the
`codex/trash-sorting-demo` work. Final PR and source commit are available in the
repository history. No merge is implied by deployment.

## Initial release: software and archive

- Initial service unit/integration tests: 16 passed. Includes input validation,
  configured-location authority, authentication, stable-UUID deduplication,
  pagination, image retrieval, separate opinions, ingest-role isolation, and
  Redis-outage startup recovery in the VLM worker. Later regressions cover
  conflicting UUID retries, closing unconsumed streams, and JPEG decompression
  bombs returning validation errors.
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

## Initial release: scene-level VLM

- Queried the live eisbox Ollama inventory: `qwen3.5:27b-q4_K_M` advertises
  vision capability. Real JPEGs were accepted by `/api/chat` and produced
  descriptions saved as separate opinions. Detection labels were unchanged.
- The initial startup worker failed if Redis was not ready; corrected and
  covered by an outage/retry regression test. Redis restart recovery was then
  exercised. Sampling skips/failures remain explicit. New responses carry
  output-limit metadata; older prototype responses predate this marker.
- This release preceded per-box readings and the evaluation workflow below.

## Public authentication

- Protected root `rgb-root` registered to `rgb-classifier/rgb-web:80`.
- Initial grant is owner-only. `/onboarding` uses separate enrollment audience
  and the existing operator approval workflow, through LaminarAuth PR36.
- Public Cloudflare DNS/tunnel/access state read back after writes. Root returns
  an Access302; request form returns200. Direct unsigned origin returns403.
  All path checks include the live/history/image protected root.
- LAN router initially cached the hostname's earlier NXDOMAIN. A scoped SIGHUP
  cache clear on its verified dnsmasq processes resolved the issue without
  restarting the router or changing configuration. Public and LAN DNS now agree.
  An existing owner Google session completed sign-in through the real public
  hostname; live overlays, archive images, and a separate Qwen opinion were
  visually verified in Chrome. No authentication bypass or forced-host mapping
  was used for that browser acceptance.
- LaminarAuth implementation suite1253 passed/37 skipped; separate reviewer
  independently ran460 passed/19 skipped and reported no actionable findings.
  PR36 CI jobs passed. Source and infrastructure receipts are retained privately
  outside this repository; no credentials or validation emails were published.
- Live denied-user, wrong-valid-audience acceptance and request/approval/welcome
  round trip have not been exercised. Owner Google access and unsigned denial
  were verified. Rollback
  instructions exist; an auth edge rollback rehearsal has not been performed.

## Uno Q primary deployment

- User explicitly approved replacing its live bird classifier. Stopped bird
  App Lab containers, image, and private app configuration remain available.
  Historical bird archive was unchanged; live viewer admission was disabled
  and the retained source mapping now points to a closed loopback port.
- Built natively on the Uno Q from board source. Image ID:
  `sha256:d3074366dc2a7c9064b76158b2197b9a220a09f8577867fc738461fad6f65d86`.
- App Lab started the new custom brick and read back
  `RGBGeneralClassifier (/home/arduino/ArduinoApps/rgbclassifier)` as default.
  Both app containers are healthy. Non-capture Venus nodes were skipped and
  USB capture selected `/dev/video2`.
- Blank-frame inference164.5ms with no detections. Live inference approximately
  160–165ms with actual TV/chair boxes, at the configured2FPS ceiling and two
  inference threads. Observed container memory102MiB and CPU67%. After several
  minutes thermal sensors were38–47.5°C, with unchanged boot ID. This is a
  short operational observation, not a long-duration thermal qualification.
- New source `unoq` observations and images reached Redis, with sampled Qwen
  opinions. Earlier `portentauno` observations remain. Location is unknown,
  deliberately avoiding an unconfirmed desk location for the replacement camera.
- Persistent outbox pending count0. The app's saved default was verified;
  a power-cycle/autostart drill for the new Uno Q application was not performed.
- Final public desktop browser acceptance includes live bounding boxes,
  observation selection, archived image/boxes, detector confidence and separate
  VLM opinion. A private screenshot is kept outside the public repository.

## Deployed service identity

- Service image: `localhost:30500/rgb-classifier@sha256:230fb9b67a878b2e80fce4ac7da6e066e99234583de623ee505642cec335961c`.
- Redis image: `redis:7.4.6-alpine@sha256:3b73847e72874be07e6657b129a94761662b79bc0f679273757d4218573b2a98`.
- Primary source:192.168.1.127, `unoq`. Ingest NodePort:pragmatic32081, restricted
  by network policy to that source. Redis and the web service remain private
  ClusterIP services. Source location is currently unknown.
- App PR:https://github.com/Wehden/RGBGeneralClassifer/pull/1.
- Auth PR:https://github.com/Wehden/LaminarAuth/pull/36.
- Both PRs remain unmerged. Final CI outcome is reported in their check runs.

## Per-detection event streams and review release

Verified 2026-10-24 evening, America/Los_Angeles:

- 36 service tests passed. Coverage includes immutable initial results,
  idempotent enhancements, confidence-dependent projections, actual failure
  response retention, crop coverage validation, live visual matching,
  settings validation/audit, human evaluation records, terminal-job replay,
  processing failures, and both export formats. Export tests cover concurrent
  updates after the captured watermark and consecutive download slot release.
- Qwen3.5 `qwen3.5:27b-q4_K_M` on eisbox received actual desk crops plus scene
  context. Observation `cab1cc2f-43de-4771-a2d7-d5e2a40677a4` completed three
  box readings in 8.65 seconds: `tv` became `computer monitor` with score0.95;
  both keyboard readings returned `keyboard` with score0.98.
- The public authenticated review of observation
  `ecb85551-8663-49bc-b1e2-203ff59626e8` displayed four completed readings,
  including `tv`0.6484 to `computer monitor` 0.95. Its history showed the initial
  result, started request, successful request taking11866.785ms, and appended
  enhancement. Original labels and scores remained in the initial record.
- Both public sliders were changed to0.46/0.66 and applied. Uno Q health read
  back revision1 with those values. They were restored to0.45/0.65 in revision2.
  Both revisions retained the authenticated operator and UTC timestamps.
- Downloaded both bundles through the protected public dashboard and opened
  every table with PyArrow. The Parquet snapshot at04:00:25 UTC contained179
  observations,404 detections,146 legacy opinions,413 detection updates,
  76 attempt records,2 setting revisions and0 evaluations. The Arrow snapshot
  at04:00:31 UTC contained180 observations,409 detections,146 legacy opinions,
  426 updates,78 attempt records,2 setting revisions and0 evaluations.
  Both contained UTC timestamp columns, typed scores and stable join IDs.
  Seven detections in each snapshot retained applied live matching values.
- No evaluation labels were inserted into the operational dataset during
  acceptance. Human evaluation persistence and exported values were exercised
  with isolated test fixtures.
- Native Uno Q image `rgb-classifier:unoq-20261024-v4` read back as
  `sha256:2fbb8cc87f95fd0326a01e80175a00a53c3cae89845442f14946a05f6245ef2d`.
  Camera capture was `/dev/video0`. A current health sample measured160.78ms
  inference. A short runtime sample measured73.42% CPU,113.2MiB memory and44.6°C.
  Health and archived observations contained enhanced live labels and their
  reference detection IDs, IoU, crop hash distance and reading age.
- Live labels use white antialiased text on dark backgrounds, with label
  positions clamped inside the image. The final live view was checked after
  the native board rollout.
- Service release image:
  `localhost:30500/rgb-classifier@sha256:81ff482d62d88424ec47563ad5e37022c194409c5152ff760aa2ca436108eef0`.
  Web and ingest use the same digest; only the web role starts the VLM worker.

## Continuous tracks and material mode

Verified 2026-10-24 evening, America/Los_Angeles:

- Investigated a monitor overlay that retained `tv` while Qwen had returned
  `computer monitor` 0.95. The boxes overlapped by0.9715, but the crop hash
  distance was20 against the former limit4. The crop-content gate was removed
  in favor of continuous track association; hash distance remains recorded.
- Uno Q health subsequently reported `label=tv`, `display_label=computer monitor`,
  `display_confidence=0.95`, `match_method=continuous_track`, hash distance15,
  IoU0.9661 and reading age53.25s on track
  `155efa51-6908-4f65-a378-0070315a6d59`.
- Qwen schema3 returns object readings and a list of material readings together.
  Actual observation `b2919206-203e-4f48-8a9e-c486fa402cd0` returned plastic0.90
  and glass0.85 for the monitor region, plastic0.95 for one keyboard region,
  and metal0.90/plastic0.85 for another keyboard region. These are recorded
  model outputs, including their individual reasons and request provenance.
- The public Materials selector saved revision5 and displayed those readings.
  Object/material mode changes are retained with threshold-setting revisions.
- Following a user-reported threshold503 during rollout, saved settings still
  contained revision2, classifier0.45/VLM0.65. A public classifier save to0.46
  succeeded as revision3 and was read back from the Uno Q. Revision4 restored
  the classifier to0.45. Recreate service rollouts have a short availability
  gap; steady-state threshold saves were exercised through the public hostname.
- 44 tests passed, adding continuous tracks, changed screen content, detection
  gaps, per-material score validation/filtering, mode auditing, and typed
  material/evaluation exports.
- Native board image `rgb-classifier:unoq-20261024-v6`:
  `sha256:5681fded520199169776313dbe6d430ec7789c9beb55375d68c910e2dc5729b9`.
- Service image:
  `localhost:30500/rgb-classifier@sha256:07170eba2e189d2d8b62253cefed99ceefd2a9542075a4957ec67ed82f53959b`.
- The detector still proposes COCO object boxes. The label map contains no fan
  class; no open-vocabulary detector has been deployed in this release.
- Final schema3 bundles were downloaded through the public page and opened
  with PyArrow. The Parquet snapshot at04:19:20 UTC contained255 observations,
  691 detections and58 material rows. The Arrow snapshot at04:19:24 UTC
  contained255 observations,691 detections and64 material rows. All eight
  tables were readable. Material rows carried individual scores, reasons,
  accepted flags and detection/attempt join IDs.
- The user's subsequent classifier0.39 change saved as revision7 and was read
  back from the Uno Q. Object mode remained selected. CI passed for the code
  commit containing material mode and continuous-track matching.

## Records access and opt-in location

- 47 tests passed. Added material/object/source/time/location query filters,
  cursor pagination without duplicated detections, typed page downloads,
  location-input validation, ingest-role isolation, location audit and
  preservation of historical location snapshots when settings change.
- Complete archive exports now use schema4 and include `location_updates` as
  a ninth table. Observations include captured location method/revision,
  coordinates and accuracy when present.
- Service image:
  `localhost:30500/rgb-classifier@sha256:23351e0f6db3d2998ea8ab9abe56135f363b74789abc8398639a24e2f7afc305`.
- Location permission is requested only by the explicit browser-location
  button. No real browser location or Google Maps request was issued during
  automated acceptance; manual/browser coordinate persistence was tested with
  isolated fixture data. Existing Uno Q location remains unassigned.
- Public `/records` acceptance filtered `source=unoq&material=plastic`, copied
  its request URL and followed its continuation cursor. Both page downloads
  opened in PyArrow with50 unique detection IDs and UTC timestamp types.
- The public schema4 complete Parquet bundle contained285 observations,
  799 detections,185 material rows,0 location updates,146 legacy opinions,
  1870 detection updates,316 attempt records,8 settings revisions and0
  evaluations. All nine typed tables were readable. The location control
  showed disabled/revision0 without requesting browser location.

## YOLOE-26n prompt-free trial and 9B pool

- Deployed native Uno Q image `rgb-classifier:unoq-yoloe26n-20261025-v2`,
  image ID `sha256:e2e455f23cc37a9589a44d87c8833155ac93cb55852af5713f9e8c474bfee873`.
- Model is the official prompt-free YOLOE-26n checkpoint, exported to float32
  ONNX at 320 px with 4,585 names, end-to-end selection, one class per proposal
  and up to 100 outputs. The graph hash is
  `8120cf9b3f3cf1dfbbc768b2eb869b48a18f6958f7272e1ba59258c1030ae806`.
  The full label mapping and export/checkpoint hashes are in its manifest.
- Capture is uncapped (`MAX_FPS=0`). Ten live health samples spanned 23 frames
  over 19.112 s: 1.2034 FPS, with 805.13 ms median reported inference. Container
  memory was 457.4 MiB, CPU 196.4% of one core, and board temperature 50.0 C.
  The initial independent profile comparison is in `yoloe-trial.md`.
- Live observation `74c84975-7920-4a84-b74a-4bd940c1bde8` recorded `office desk`,
  `cup` and `computer chair`, with detector/vocabulary provenance. Public review
  showed these initial YOLOE readings and their scores.
- 66 Python tests and 7 browser behavior tests pass. Added letterboxing/channel
  order, clipping, confidence/class validation, mask-column handling, pinned
  graph/vocabulary, bounded configuration validation and typed provenance tests.
- Both schema6 complete formats opened successfully at 690 observations,
  including 6 YOLOE observations. Parquet and Arrow page downloads retained
  detector model/hash, initial class ID and vocabulary hash. Older EfficientDet
  observations remained available.
- RGB's new target is the 9B pool, model `qwen3.5:9b-q4_K_M`, with 8K context.
  A bounded image/crop driver probe returned HTTP 200 in 6,934.75 ms and a valid
  object/material response. Model inventory reported the expected vision model.
  The probe retained its full transcript privately and did not advance the
  analysis queue. VLM processing remained paused and idle throughout rollout.
- The public dashboard identified both `yoloe-26n-pf-320` and the 9B model.
  Namespace/label-selected egress reaches the pool. No pool membership, 27B
  deployment or other consumer configuration was changed.
- Service image:
  `localhost:30500/rgb-classifier@sha256:f97c70ade5bc120adea81d9164447199a00ec839ea1e4606fdae830f4335e8d2`.
- The v7 EfficientDet image and compose rollback file remain on the Uno Q;
  no historical observations or rollback images were removed.
