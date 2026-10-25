# Design and contracts

The approved goal is a usable board demonstration with live bounding boxes, a
protected public feed and event archive, and portable deployment on future
Seattle University k3s nodes. Birdwatcher on Uno Q and all historical bird events
must remain intact. The X8's old containers and the Uno Q's bird app/image remain available for rollback.
The user subsequently approved repurposing the existing Uno Q as the primary
RGB camera; its live bird workload is retired and historical bird events remain.

## Observation path

The board probes actual V4L2 capture devices and owns the selected camera once.
EfficientDet-Lite0 runs on each frame; live JPEGs include normalized boxes,
labels, and confidence. Up to four board viewers share the same frame cache.
Health and snapshot endpoints use cached state and do not open another camera.
The cluster admits up to six live viewers and reserves separate worker capacity
for history and health requests. Board limits can return HTTP 429 sooner.

While objects are detected, the board emits one scene observation every fifteen
seconds. Each observation may include several objects. Repeated observations
can show the same physical item. This is not a unique-item counter, a tracker,
or a quantitative waste audit. A persistent SQLite outbox holds up to 500
observations and retries delivery with stable UUIDs. When full it logs and skips
new observations, preserving already queued observations. It does not promise
unbounded offline capture.

The ingest-only NodePort accepts only POST `/api/ingest` and health. All history,
images, static files, and live routes return 404 on that role. A constant-time
bearer-token check precedes validation. Network policy permits this port only
from the configured board address. Redis remains ClusterIP-only with a password
and pod-scoped network policy. Public web and live routes share the protected
LaminarAuth root ingress. The `/onboarding` routes use the existing enrollment
service and operator approval. Students are not automatically granted access.

## Event schema and persistence

An event contains a UUID, UTC observed/received times, source ID, configured
location (or unknown), model name and SHA-256, detection list, image SHA-256,
separate model opinions, and `sorting_status=unassessed`. Detection coordinates
are normalized `[ymin,xmin,ymax,xmax]`. The server validates scores, boxes,
source allowlisting, timestamps, model provenance shape and bounded JPEGs.
The server takes location from source configuration, ignoring board claims.
No GPS information is synthesized.

Images are written and fsynced before an atomic Lua script deduplicates UUIDs
and appends to `rgb:events`; `rgb:ids` maps UUIDs to stream IDs. A retried UUID
returns its existing entry. Payload conflicts cannot replace the event or its
image. A crash between image write and Redis append can leave an orphan image.
No automatic trimming or deletion occurs. Operators must monitor disk and Redis
memory; `noeviction` refuses new events rather than silently dropping history.
At Redis's configured 512 MiB memory ceiling, board delivery retries and its
outbox eventually fills. The 25 Gi PV capacity is a Kubernetes accounting value,
not an NFS filesystem quota.

Redis uses AOF with `appendfsync everysec` and periodic RDB snapshots. The final
second of acknowledged writes may be lost on a server crash. An intact board
outbox protects unacknowledged deliveries, not all acknowledged events. Shared
NFS data can follow a pod to a different node, but standpunkt is a single storage
server and Redis is one process. This deployment is not highly available.
Shared filesystem storage alone is not replication or a tested backup.

## VLM extension

Optional `VLM_ENDPOINTS_JSON` entries identify exact endpoint/model pairs. A
single worker handles one event at a time and records separate responses,
timestamps, or explicit failures; it never overwrites detector labels. The
cursor and opinion list commit in one Redis transaction. Responses are treated
as plain untrusted text in the browser. Image text has no authority over tools
or infrastructure. VLMs receive no tools, credentials, or disposal decision
permission. Multiple entries provide distinguishable opinions, not a calibrated
quorum or verified consensus.

The initial model is the live-verified Laminar Ollama Qwen3.5 27B model. Vision
capability and inventory are checked before enabling it. An outage leaves live
capture and event ingestion operational. The VLM worker samples at most once per sixty seconds (configurable via
`VLM_MIN_INTERVAL`), skips observations older than five minutes, and records
explicit skip reasons. It may lag the archive;
it has one bounded request at a time per endpoint, with a ninety-second read
timeout. Images stay within the configured cluster endpoint.
