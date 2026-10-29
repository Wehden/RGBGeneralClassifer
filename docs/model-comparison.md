# Per-box model comparison

The local console's **Per-box model comparison** panel shows a retained
RGB snapshot with numbered boxes and both model readings for each scored
box. YOLOE supplies the object label and detection confidence; SigLIP2
supplies the material candidate, native score and identification decision.

These scores concern different tasks. The view does not subtract, average,
rank or interpret their unlike labels as a disagreement. The deployed
material head's learned score also remains distinct from a legacy
image/text cosine-similarity score.

## Exact pairing

The material worker receives the original detector metadata with its
immutable camera frame and boxes. Every fresh material reading retains a
`paired_object` record with the original YOLOE label, confidence, model hash,
frame ID and bounds. The snapshot contains only freshly scored rows from
that pass; a cached material result from an earlier frame cannot replace one.

The live overlay may still reuse recent track readings for responsiveness.
The comparison panel displays its own scored snapshot so those reused
readings cannot make a different live frame look like an exact comparison.
Archived material JSON retains the original paired object reading, and
the capture review shows its confidence alongside the material score.

The existing crop worker processes up to four eligible boxes per pass.
Freshness limits may reduce that count. The panel reports how many boxes
were scored out of the frame's total. It runs with automatic capture paused,
uses bounded memory, and requires no additional VLM call or observation.

## Camera API

`GET /api/camera/comparison` returns the latest matched snapshot:

- Frame/session IDs, capture time, calibration hash and image hash.
- Original detector metadata and `pairing: "exact_frame_and_box"`.
- `same_task: false`, total box count and freshly scored rows.
- Each row's box index, track, bounds, crop hash, object reading and material
  candidate, score type, margin, candidates and acceptance policy.
- Snapshot age and `complete` or `stale` status; snapshots older than five
  seconds are marked stale.

Use the returned `image_url` to fetch
`GET /api/camera/comparison/image.jpg?frame_id=<UUID>`.
The preview preserves the source frame's aspect ratio with a maximum long
side of 768 pixels. Its response includes matching frame/session, capture
time, calibration and image-hash headers. Both responses disable caching.

Only the two latest scored images are retained. An expired frame ID returns
410; it never substitutes a newer image. Camera-session or calibration
changes hide previous snapshots and prevent an old scoring pass from
publishing into the new context. This retention is separate from the live
camera frame cache and the observation archive.

## Console behavior and verification

The browser loads the frame-qualified image before committing the image,
numbered boxes and score table together. Source switching and closing the
panel cancel pending work. Requests and image loads have bounded timeouts;
an image-expiry race triggers a refresh while the previous matched view
remains intact. Open panels refresh approximately every two seconds.

Verification covers:

| Claim | Check |
|---|---|
| Object/material readings use the same original frame and box | Pairing regression, including cached reuse on a newer live frame |
| Comparison creates no observations or cloud calls | Read-only preview regression |
| An expired image cannot silently become a newer one | Two-image retention and 410 regression |
| Session/calibration resets discard old comparisons | Published-snapshot and in-progress-reset regressions |
| Old captures remain explicitly dated | Stale-snapshot regression |
| Image and table appear together | Browser-controller commit regression |
| Switching source cannot commit late scores | Cancellation regression |
| Learned and cosine score types remain distinct | Browser score-format regression |
| Camera resets remove the previous image and table | Empty-snapshot browser regression |

The application suite passed with 284 Python tests and one skipped trainer
module in the application environment. All 36 JavaScript tests passed.
Live API checks verified image hashes and exact frame/box pairing; a real
browser checked image loading, table rendering and collapse/reopen behavior
without page errors. The owning Qt console was restarted with the new panel,
and its saved settings, including paused capture, were preserved. The camera
service remains bound to loopback.

The first deployed run's live API checks, settings receipts and browser
screenshot are retained privately under
`~/.local/state/bolt-rgb/model-comparison-2026-10-28/`.
