# Operations and lab portability

## Current home deployment

The primary camera is now the existing Arduino Uno Q (`birdwatcher`, currently
192.168.1.127), repurposed with the user's approval. It runs a native ARM64
build of `board/Dockerfile` inside an app-scoped Arduino App Lab custom brick.
The cluster remains responsible for Redis, image storage, the web dashboard,
and optional VLM inference. No vision-model download occurs during startup.

The Portenta X8 at 192.168.1.231 is paused for the user's active-cooling work.
Its RGB images, runtime file, outbox, and original bird rollback containers are
retained. Do not restart the unbounded X8 experiment during this work. The
current board code defaults to two LiteRT threads and at most two FPS; tune
`INFERENCE_THREADS` and `MAX_FPS` only after measuring sustained temperatures.

Source locations are deliberate operator configuration. Historical X8 events
say `Home desk`, matching the user's identified camera placement. Uno Q's
location is currently unknown until its new camera placement is confirmed.
Change `SOURCES_JSON` in `deploy/app.yaml`, then apply and restart web/ingest.
Changing the board's `SOURCE_LOCATION` alone does not change archived location;
the service's configuration is authoritative.

## Build and release

From the repository root, build the web service for the target architecture,
push it to the verified registry, and retain its immutable digest. Home workers
pull through `localhost:30500`; the current push endpoint is
`laminarflow:30500`. These are home-specific registry conventions.

```sh
docker build -t laminarflow:30500/rgb-classifier:RELEASE .
docker push laminarflow:30500/rgb-classifier:RELEASE
python3 scripts/render-home.py REGISTRY/rgb-classifier@sha256:DIGEST --output /tmp/rgb-app.yaml
kubectl apply -f deploy/namespace.yaml
# Provision the dedicated NFS directory/export once; review addresses first.
# deploy/storage-provision-home.yaml is a privileged, one-shot host setup pod.
# Remove it after successful provisioning. Never run against another app's data.
kubectl apply -f deploy/storage-home.yaml
# Supply private rgb-runtime Secret before starting workloads.
kubectl apply -f deploy/network-policy.yaml
kubectl apply -f /tmp/rgb-app.yaml
kubectl -n rgb-classifier rollout status deploy/rgb-web
kubectl -n rgb-classifier rollout status deploy/rgb-ingest
```

The `rgb-runtime` Secret needs `REDIS_PASSWORD`, a `REDIS_URL` using that
password and private `rgb-redis:6379/0`, and a random `INGEST_TOKEN` of at least
32 characters. Generate credentials privately, preserve them across releases,
and never commit or print them. Redis is ClusterIP-only. The ingest role alone
is exposed at the home NodePort 32081 on pragmatic, with local external traffic
policy and network policy restricted to the Uno Q's address. History and live
routes return 404 on this role. Recheck those restrictions after changing nodes
or CNI/network-policy implementations.

## Uno Q installation and rollback

Build the board image natively from repository contents:

```sh
docker build -f board/Dockerfile -t rgb-classifier:unoq-RELEASE .
```

Copy `deploy/unoq/` to `/home/arduino/ArduinoApps/rgbclassifier`, set the custom
brick's image to the tested native image, and provision a private
`/home/arduino/rgb-classifier/runtime.env` (mode 0600) plus an owner-writable
`data/` directory. Use `SOURCE_ID=unoq`, the cluster's ingest token,
`ARCHIVE_URL=http://192.168.1.94:32081`, and `OUTBOX_DB=/data/outbox.sqlite3`.
The current native image tag is recorded in the verification receipt.

Before switching, preserve the old bird App Lab directory and image. Stop the
old app with `arduino-app-cli app stop ~/ArduinoApps/unoclassifier`, start the
new app with `arduino-app-cli app start ~/ArduinoApps/rgbclassifier`, and make
it default with `arduino-app-cli properties set default
~/ArduinoApps/rgbclassifier`. Read back the default property. The platform
requires `python/main.py`; inference belongs to the custom brick, which alone
publishes port 8080. Do not add app-level port publishing. The compose fragment
maps video0 through video3; the detector probes actual capture devices and
skips non-capture Venus devices. Revalidate nodes and video group 44 when the
camera or board OS changes. The existing powered hub must remain attached.

The previous bird app can be restored by stopping RGB, setting the old app as
default, and starting it again. Its image and private configuration remain on
the Uno Q. App Lab may recreate its stopped containers from that retained image.
The historical bird archive was not moved or deleted. Its live viewer admission
is deliberately disabled (`MAX_LIVE_VIEWERS=0`), and its retained `birdwatcher`
source points to a closed loopback port rather than the repurposed camera.
That mapping keeps historical source identity available without relaying RGB
as a bird stream. To roll back live birds, restore the previous mapping to
`http://192.168.1.127:8080`, restore viewer admission, restart the archive app,
and verify actual bird emission. Do not restore the retired X8 source unless
it is intentionally returned to bird duty.

## Health, limits, and recovery

Check the board `/healthz`, camera freshness, inference time, the outbox row
count, cluster event count, and retrievable image bytes. Kubernetes readiness
alone does not establish camera health. Stop a camera stream before moving a
USB hub; changing hub power can restart these boards. No forced board power
cycle is part of automated rollout.

Board viewer limit is four; the cluster reserves worker capacity with its own
six-viewer cap. Disconnects release upstream sessions. Live failures do not
prevent browsing history. The outbox retries after network/storage outages.
No automatic event expiry occurs; monitor Redis `used_memory`, NFS free space,
and board outbox usage. The NFS PV is Retain and shared across nodes, with a
single standpunkt server. Its 25 Gi capacity declaration is not a quota.

VLM configuration accepts a JSON list of endpoint/model pairs. The home
endpoint is `ollama-vlm-pool.ollama.svc.cluster.local:11434` with
`qwen3.5:9b-q4_K_M`, using an 8K context. The previous 27B endpoint remains
available to other consumers. Every archived box
receives a model reading and comparison, in batches of up to four crops plus
scene context. Threshold controls are in the dashboard; revisions are audited
and the board retrieves them every three seconds. Review downloads provide
Parquet or Arrow IPC tables. See [review data](review-data.md) for schemas,
request/evaluation provenance and the live-label matching rule. An empty model
endpoint list disables automatic analysis.

## Backups and restoration

A Redis restart and an isolated RDB restore have been exercised; there is no
scheduled backup installed by this project. For a backup, request `BGSAVE` with
the private authenticated client, wait for `rdb_bgsave_in_progress=0` and a
successful status, then copy the completed `dump.rdb` and the referenced image
files to separately managed storage. Preserve UUID indexes, opinions, and VLM
cursor with the event stream. Snapshotting only images or only Redis does not
produce a complete archive. Keep private photos out of the public repository.

Restore into an isolated Redis instance first, verify event/index counts and
sample image hashes, and only then change production storage. AOF may take
precedence over RDB on startup; choose the restore mode deliberately. Shared
NFS and a PV Retain policy do not protect against server loss or deletion.

## Seattle University k3s on AGX Thors

The service has no GPU dependency. Build/push a native ARM64 service image on a
Thor or publish a tested multi-architecture image. Remove the home `amd64`
selector from web and replace the ingest node selector with the chosen lab
node. Choose an unoccupied NodePort and point the board at that node; update
source IPs and network policies. Do not reuse home IPs, tokens, VLM endpoints,
NFS exports, or a home source location.

Use the lab's actual shared/distributed storage class for a new RWX PVC, or
provision a dedicated shared NFS export. Replace `storage-home.yaml`; initialize
separate `redis/` and `images/` directories owned by UID1000. If the lab only
has local-path, explicitly accept its node binding and absence of HA before
using it. Do not label a collection of Thor host disks distributed storage.

Provision new private credentials and use the same event contract. VLMs are
optional; leave the endpoint list empty until a real lab vision model has been
queried successfully. Republish the tested CPU detector to the board; it does
not need CUDA or Jetson-specific packages. A USB camera remains mandatory.

For an internet-facing lab site, register a deliberately chosen hostname and
origin through LaminarAuth, including its separate enrollment audience,
protected root, identity strip before forward-auth, default-deny policy, DNS,
and tunnel. Prove unsigned denial and authorized Google access for live,
history, and images. Moving the existing `rgb.wehden.com` origin is a separate
cutover decision; do not silently route the home hostname to a lab cluster.
