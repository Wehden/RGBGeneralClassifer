# Local Femto Bolt and Bedrock operation

## Install and start

Use Python 3.12 on Linux with USB access to the attached Bolt:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
.venv/bin/python -m rgb.local
```

The official distribution is `pyorbbecsdk2`; Python imports it as
`pyorbbecsdk`. Its wheel includes OpenCV. Use the local requirements in a clean
environment rather than installing both OpenCV distributions together.
The pinned YOLOE ONNX artifact and manifest are already in `models/`.
Startup does not download model weights or change camera firmware.

The default dashboard/API address is `http://127.0.0.1:8080`; camera access
does not require authentication. The default archive is
`~/.local/share/bolt-rgb`, containing SQLite WAL files and content-addressed
sensor/request assets. Physical USB serial and resolved archive identity
are locked independently in a common per-user runtime directory. This prevents
another local runner, a different state directory, or a V4L2 alias from acquiring
the same camera. The archive lock precedes recovery.

AWS calls use the normal SDK credential/refresh chain for profile `default`,
source region `us-west-2`. The default inference profile is
`us.amazon.nova-2-lite-v1:0`; reasoning is disabled. Credentials are not copied
into application state. Authentication, permissions, throttling, and timeouts
appear as recorded request failures.

## Controls and configuration

An X11-compatible Qt console is also available as `python -m rgb.desktop`.
It shares the dashboard and local service and adds desktop menus and native
save dialogs. See [desktop operation](desktop-console.md).

The dashboard provides Color, registered Depth, and native IR views. Depth
preview uses TURBO over 0–5 metres, with invalid pixels black and values above
5 metres clipped. IR is percentile-normalized reflectance, not temperature.
Raw uint16 planes remain unchanged. Color and registered-depth coordinates
match; native IR uses its own camera coordinates.

The view selector also provides a **streaming 3D RGB-D point cloud**. Drag to
orbit, Shift/right drag to pan, scroll to zoom, and double-click to reset.
RGB shading uses the color pixel associated with registered depth; depth
shading uses blue near points and orange far points. Point coordinates use
metres, calibrated color-camera rays and registered depth. Invalid depths
and points beyond the viewer's 5-metre range are omitted. This is a live
point cloud, not a fused mesh or a metrology certification.

Automatic observations now default to **one per 60 seconds**, including the
frames used for automatic VLM analysis. The previous separate live VLM
capture path could save a new frame for every model job; that path now reuses
the sampler's archived frame. A scene can still have multiple detections,
history entries and VLM batches. The interval limits new automatic scenes,
not individual database journal rows.

Set **Capture every … seconds** and click **Apply interval** to save an interval
from 1 to 86,400 seconds. **Pause capture** stops new automatic observations
and undispatched live VLM work while the camera, detector and all live views
continue. An already-started model request finishes and keeps its audit
record. Manual analysis of existing observations remains available.
**Capture once** is an explicit one-shot save, including while automatic
capture is paused. Resuming samples the current scene without replaying a
backlog. The interval and capture pause survive restarts.

The model selector supports explicit Nova Lite fallback and Nova 2 restoration.
In-flight requests retain their model. Queued manual analysis retains its saved
policy. A policy change supersedes remaining live batches, and new live work
uses the new policy. Invalid, truncated, or incomplete readings remain failures
and cannot update labels. There is no automatic model switch or SDK retry.

**Nova Lite had lower full-schema reliability in the measured comparison.**
See the verification results before relying on it for this workload.

**Pause VLM** allows an in-flight request to finish and prevents new dispatch. Camera,
archive, and review continue. Saved model, pause, thresholds, and display mode
survive restart. Explicit CLI values override application environment values,
then saved values; implicit defaults do not reset saved choices.

```sh
.venv/bin/python -m rgb.local --aws-profile default --aws-region us-west-2 \
  --camera-serial CL85763022X --port 8080

# Camera, archive and review, without any cloud inference:
.venv/bin/python -m rgb.local --no-vlm

# Normal service, initially paused, including manual jobs:
.venv/bin/python -m rgb.local --paused

# Deliberately choose fallback, or restore primary:
.venv/bin/python -m rgb.local --model us.amazon.nova-lite-v1:0
.venv/bin/python -m rgb.local --model us.amazon.nova-2-lite-v1:0
```

`--camera-serial` is needed if multiple eligible Bolts are present.
An explicitly selected V4L2 fallback provides color only:
`--camera-backend v4l2 --camera-device /dev/video0`.
Depth/IR endpoints then return 404. V4L2 aliases resolve to the physical USB
serial for ownership.
If the SDK has detached a V4L2 kernel node, specify `--camera-serial` to identify
the intended physical camera. An unidentifiable node fails startup explicitly;
a raw Y16/Z16 depth node is not accepted as a color camera.

Run `--help` for timing/profile options. Application environment names include
`RGB_HOST`, `RGB_PORT`, `RGB_STATE_DIR`, `RGB_AWS_PROFILE`, `RGB_AWS_REGION`,
`RGB_CAMERA_SERIAL`, `RGB_CAMERA_BACKEND`, `RGB_CAMERA_DEVICE`, `RGB_MODEL`,
`RGB_VLM_PAUSED` (`true`/`false`), `RGB_DISPLAY_MODE`,
`RGB_DETECTOR_THRESHOLD`, `RGB_VLM_THRESHOLD`, `RGB_CAPTURE_FPS`,
`RGB_DETECTOR_FPS`, `RGB_ARCHIVE_INTERVAL`, `RGB_VLM_MIN_INTERVAL`,
`RGB_READ_TIMEOUT`, `RGB_CONNECT_TIMEOUT`, `RGB_MAX_TOKENS`,
`RGB_MAX_SKEW_MS`, and `RGB_FRESHNESS_S`.

VLM dispatch within a sampled observation has no one-Hz ceiling. New automatic
observations follow the saved capture interval. The default detector target
is 5 FPS and the default synchronized sensor profiles are 15 FPS. These are
independent of Bedrock latency. Status reports delivered capture rate,
detector time, pending replacement counts, model outcomes, and workload-specific
request latency. Latency windows hold 60 successes per workload, with at most
128 recent workload groups. Fewer than 20 samples is provisional evidence.

## Direct camera API

These calls require no credentials and create no model requests:

```sh
curl -s http://127.0.0.1:8080/api/camera/status
curl -s http://127.0.0.1:8080/api/camera/calibration
curl -s http://127.0.0.1:8080/api/camera/detections
curl http://127.0.0.1:8080/api/camera/color/snapshot.jpg -o /tmp/color.jpg
curl http://127.0.0.1:8080/api/camera/depth/raw.png -o /tmp/depth.png
curl http://127.0.0.1:8080/api/camera/ir/raw.png -o /tmp/ir.png
curl http://127.0.0.1:8080/api/camera/frame -o /tmp/bolt-frame.npz
```

`/{color,depth,ir}/stream` provides MJPEG, with eight concurrent viewers across
channels. JPEG snapshots and stream parts identify their frame/session and
host receipt time. Depth snapshots are registered previews; depth raw PNGs are
native uint16. `X-RGB-Plane` and scale headers distinguish the planes.

Snapshot, raw PNG, calibration, frame bundle, and detections accept
`?frame_id=<UUID>` to pin an entry in the memory cache. Malformed IDs return
400; evicted IDs return 410; a supported channel without fresh data returns
503; unsupported channels return 404. A pinned frame without a matching
detector result returns 202. Unpinned detections identify both their processed
frame and the newest camera frame.

The NPZ can be read with `numpy.load(..., allow_pickle=False)`. It contains
`color_rgb` (uint8 RGB), `depth_native`, `ir_native`, `depth_registered`
(uint16), and `metadata_json_utf8` (uint8 UTF-8 JSON). Metadata records channel
timestamps, actual profiles, session/sequence, native and registered depth
scales, and canonical calibration.

Archive a frame independently of VLM:

```sh
curl -s -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8080/api/camera/observations
```

Without `frame_id`, capture selects the latest fresh detector-completed frame
from the current camera session and calibration, including its matching sensor
evidence. If no such frame is ready, it returns HTTP 503 without saving an
observation; retry after detection completes.

Use `{"frame_id":"<UUID>"}` to pin a cached frame. A new pinned observation
requires a matching completed detection result; otherwise it returns HTTP 503
without saving. Repeated capture of an already archived frame
returns its original observation ID. Sensor assets are available at
`/api/events/<observation-id>/assets/{depth_native,ir_native,depth_registered,depth_preview,ir_preview,calibration,metadata}`.

### Point cloud API

`GET /api/camera/pointcloud` returns a bounded binary frame without invoking
Bedrock or creating archive records. Optional parameters are `frame_id`
(the same pinned-frame contract), `stride` (2–32, default 6) and `max_range_m`
(greater than zero through 20, default 5). Sampling may increase the stride
to keep at most 30,000 points. Two concurrent conversions and two cached
payloads bound the conversion work and memory. A color-only V4L2 backend
returns 404; stale data or unusable calibration returns 503.

The first four bytes are the little-endian uint32 JSON-header byte length.
The UTF-8 header follows, then `point_count` records of 16 bytes each:
three little-endian float32 values X/Y/Z in metres, three uint8 RGB bytes,
and one zero padding byte. Axes are right/down/forward in the color-camera
coordinate system. The header identifies the frame, session, receipt time,
calibration hash, sampling and source planes. Range filtering limits optical
Z depth, rather than Euclidean distance from the camera.

The viewer requests latest frames at up to 4 Hz, with one request in flight.
It cancels requests/polling when switching away. WebGL uses the local GPU;
a CPU canvas renderer remains available without WebGL. No external scripts
or remote renderer are used.

## Review and evidence

The existing dashboard and Records page work against SQLite, preserving stable
detection IDs, history, confidence projections, pagination, evaluations, and
typed Arrow/Parquet exports. A changed candidate invalidates a stale evaluation
with 409; threshold/mode changes preserve the displayed review context.
Location is opt in. Local caller identity is recorded as unknown.

`GET /api/review/export/{parquet,arrow}` creates a consistent snapshot across
the review tables. Schema 7 adds frame/session/calibration references, typed
depth summaries, request IDs, and sensor/input references. Exports use
watermarks and a SQLite read transaction while acquisition/writes continue.

`GET /api/review/evidence?event_ids=<UUID>,<UUID>` includes selected observations,
attempt metadata, original sensor assets, calibration, and exact dispatched
JPEGs. SHA-256 joins appear in the manifest. One full/evidence export is
admitted at a time.

Every paid request archives its observation and exact ordered inputs first.
Validated terminal results, enhancements, and batch progress commit together.
Interrupted requests are recorded as interrupted on restart. Never-started
manual jobs survive restart. Explicit reanalysis creates a new job/attempt with
a recovery link; interrupted paid work is not invisibly retried.

Archives have no automatic retention/deletion. Automatic observations now
share one capture interval; explicit snapshots add intentional records.
Monitor asset bytes, write rate, and
free disk through VLM status. Stop the process before manually moving an
archive; preserve its database and assets together.

## Verification

```sh
.venv/bin/python -m pytest -q
node --test tests/js/*.test.cjs

# Read-only capture/API observation; creates no Bedrock requests:
.venv/bin/python scripts/verify-local.py --seconds 900 --viewers 8 \
  --output "$HOME/.local/state/bolt-rgb/capture-check.json"

# Explicit paid comparison using a new private archive:
.venv/bin/python scripts/benchmark-bedrock-local.py --invoke-bedrock \
  --state-dir "$HOME/.local/state/bolt-rgb/model-check"
```

The model benchmark requires a running three-channel service with detections.
It replays each retained crop batch through both models, records every outcome,
and does not alter the service's model setting. It measures protocol validation,
latency, and tokens; it does not establish recognition accuracy.
