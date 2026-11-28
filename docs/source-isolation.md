# Exclusive camera source policy

The remote dashboard source selector now saves `active_source` through
`POST /api/settings`, with `X-RGB-Request: settings` and a JSON body containing
only `{"active_source":"<configured source ID>"}`. IDs must exist in
`SOURCES_JSON`; labels such as Z CAM are not policy identifiers. The local
camera selector and its device ownership contract are unchanged.

Selection is installation-wide, shared by dashboard and ingest processes
through Redis `rgb:settings`. It uses the existing atomic revision and audit
stream, including operator identity. The last successful selection wins.
Opening a page does not change selection. Reload restores the saved source;
another already-open page must reload to display another operator's selection.
A source removed from configuration stays selected in storage and disables
other sources until an operator explicitly selects a configured source.

Both `/api/board-policy/<source>` and `/api/device/<source>/policy` return
source-effective settings. When the requested source differs from
`active_source`, `classification_running` and `record_observations` are false.
The saved global flags, thresholds, display mode and VLM settings are preserved.
Selecting Uno Q intentionally restores its saved flags, including a saved pause
or preview-only mode. Global Start cannot override an inactive source mask.

Both observation endpoints reject an inactive source with HTTP 409, even for
an event captured before selection changed. Admission is checked again in the
same Redis operation that appends the scene, closing the concurrent switch
race. Rejection does not create scene, analysis or detection records. In the
narrow race after image staging, an unreferenced content-addressed JPEG may
remain; existing shared images are never deleted. Existing archives, image
access, manual archive analysis and authentication boundaries are unchanged.
Live preview transport remains available; this policy disables classification
and recording, not camera power or preview capture.

## Rollout contract

Existing installations without `active_source` retain their previous policy
until an explicit selection. The remote UI displays “Select active camera…”
in that case and does not silently choose or activate Uno Q. Before accepting
Z CAM isolation after rollout, explicitly select its configured source and
verify the saved revision and Uno Q's effective false/false policy. No migration
or source choice is performed by merely deploying or loading this code.

Deploy matching dashboard and ingest code against the same Redis. Every
writer must use this admission check; mixed old/new ingest replicas do not
provide the guarantee. The repository's current main has no Z CAM-specific
integration. Reconcile this change with any newer deployed Z CAM integration
before rollout, retaining the configured source ID and selector behavior.

The remote board runner now requires a fresh policy lease. It polls every three
seconds and fails closed 15 seconds after dispatch of the last successful,
valid policy request. The bound matches the publisher's existing 15-second
freshness window and allows several normal poll intervals. Monotonic time
avoids wall-clock changes; slow, invalid and failed responses do not extend
an enabled lease. Startup is disabled until a valid policy arrives. A fresh
response resumes only its current source-effective flags, so reconnecting
while Z CAM is selected keeps Uno Q disabled. Explicitly selecting Uno Q
permits resumption after the next valid policy response.

The lease applies only to `board/main.py`, which always uses remote Transport.
The standalone desktop/local runtime is unchanged. The board checks authority
before inference, after inference and before inserting an observation. Work
already executing cannot be interrupted, but results from an expired/disabled
policy are suppressed. Preview capture and the existing outbox remain intact.
Board health/live status includes `policy_fresh` and `policy_max_age_s`.

Offline stopping requires rolling out the updated board runner and its new
`board/policy.py` module as well as the matching dashboard/ingest release.
Until that board rollout, an old board can retain enabled policy indefinitely
while disconnected, although the updated server rejects inactive uploads.
Even with the update, stopping is bounded by the remaining lease and completion
of already-running inference; this is not immediate physical camera shutdown.
Verify the policy revision, false classification/recording flags and lease
expiry on the actual board before declaring rollout acceptance. No device
activation or deployment has been performed for this patch.

HTTP 409 deliberately does not acknowledge delivery: the board retains its
outbox and retries. Selecting Uno Q later permits queued observations under
the existing capture-time global recording policy.

Rollback to older server code removes isolation enforcement even if the Redis
selection remains. Preserve Redis and archive storage during rollout/rollback.
