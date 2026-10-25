# Detection records and review data

Each archived camera observation contains an image and a set of detections.
Each detection has a deterministic UUID derived from the observation UUID and
box index. Its initial classifier label, score, box and model identity remain
in the initial record. Subsequent model attempts, enhanced readings and human
evaluations append to that detection's Redis stream.

The displayed value is the latest enhanced label meeting the configured VLM
threshold. The initial label and every enhanced candidate remain available in
the history and exported tables. Changing the VLM threshold recomputes the
current display without deleting candidates or their scores. Changing the
classifier threshold affects subsequent board detections. Both threshold
values and each setting revision are recorded. The board polls settings every
three seconds and includes the threshold/revision in captured observations.

Qwen receives the full image and numbered crops, in groups of up to four.
Each crop returns a label, confidence, comparison (`agrees`, `refines`,
`disagrees`, `uncertain`) and visual reason. The application validates exact
box coverage, indices, scores and fields. Every new archived detection is
submitted; the former one-minute sampling interval has been removed. Older
observations can be submitted with **Analyze detections**.

Each request has an attempt UUID. A `started` record is persisted before the
HTTP request. The terminal record contains the raw response, timing, token
counts, HTTP status, parser/request outcome, model identity, endpoint, observed
model inventory, prompt text/hash, implementation hash and settings. Attempts
that return errors retain their responses and error information. Processing
failures append separately. A completed job is not reissued after a cursor
checkpoint interruption; a new requested analysis creates a new job.

Live labels use completed readings joined by a board-assigned `track_id`.
Tracks use one-to-one, same-class box association at IoU at least0.5 between
successive frames. A detection gap longer than3s ends the track. Restarting
the board creates new track UUIDs. A changing screen image no longer rejects
the reading: crop hash distance, box overlap and reading age remain recorded
measurements, rather than label-admission gates. Captured `live_value` retains
the source detection/observation IDs, selected display mode, actual displayed
label/score, object reading, materials, matching method and measurements.

## Object and material modes

The dashboard's **Display** selector switches between Objects and Materials.
The mode is a saved, audited setting and reaches the board through its normal
policy poll. Object and material readings are produced together by Qwen for
each new archived box. Material readings contain an open-text label, confidence
and visual reason; multiple materials can be recorded for one box. Material
mode displays readings individually meeting the VLM threshold, joined with `+`.
The overlay's combined score is the minimum score among the displayed materials;
the review and exports retain each individual score.

Material mode does not infer a material from a COCO object label. A legacy
reading without materials is shown as not read; Analyze detections obtains
new object and material readings. Model scores below the display threshold
remain available in review and export. The current region proposal stage is
still EfficientDet's COCO detector; this mode does not add regions for objects
that detector missed.

## Evaluation

The review pane exposes the initial label, enhanced label/score, comparison,
visual reason, request outcomes and append-only history. **Evaluate detection**
records a reference label, separate classifier/VLM correctness outcomes and
notes and the selected display mode. Each evaluation snapshots the exact initial and enhanced values and
threshold revision, with timestamp and the authenticated identity when supplied
by LaminarAuth. A later evaluation appends another record.

## Downloads

The dashboard provides ZIP bundles of Parquet or Arrow IPC tables. Both formats
preserve column types, nullable values, UTC timestamps and join IDs. Tables:

| Table | Row meaning |
|---|---|
| observations | Initial camera observation and image/model provenance |
| detections | One detection, track ID, initial score, enhanced value, display mode/value, and live matching values |
| material_readings | One material reading per current detection candidate, including score, reason, threshold result and model/attempt IDs |
| detection_updates | Every initial/attempt/enhancement/evaluation/processing-failure update |
| model_attempts | Started and terminal request records, joined by attempt_id |
| settings | Threshold revisions and operator identity |
| evaluations | Recorded reference labels and correctness outcomes |
| legacy_opinions | Previously retained scene descriptions and sampling outcomes |

`manifest.json` records schemas, row counts, export time, settings and stream
watermarks. Watermarks and settings are captured in one Redis transaction.
Detection projections are reconstructed from updates up to those watermarks.
Writers use bounded batches and a temporary on-disk projection index. One export
runs per web process. Image files are referenced by SHA-256 and are not included
in the bundle.

```python
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

# After extracting the chosen bundle:
detections = pq.read_table('detections.parquet').to_pandas()
attempts = pq.read_table('model_attempts.parquet').to_pandas()
reviews = pq.read_table('evaluations.parquet').to_pandas()

# Or read the Arrow IPC bundle:
with ipc.open_file('detections.arrow') as reader:
    detections_arrow = reader.read_all()
```

Join evaluations to detections by `detection_id`. Join enhanced readings to
request traces by `attempt_id`; pair started/terminal request rows using that
same ID. `value_json` retains complete records alongside typed columns.
Request failures, parsing failures, comparison disagreements and human-marked
incorrect results are distinct values in these records.
