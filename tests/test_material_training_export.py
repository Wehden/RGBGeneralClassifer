import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "material_training_export",
    Path(__file__).resolve().parents[1] / "scripts/prepare-material-training.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_exact_duplicates_preserve_later_session_and_its_scene_evidence(tmp_path):
    archive = tmp_path / "archive"
    assets = archive / "assets"
    assets.mkdir(parents=True)
    connection = sqlite3.connect(archive / "archive.sqlite3")
    connection.execute("create table observations (value text)")
    for index in range(3):
        path = assets / f"{index}.jpg"
        Image.new("RGB", (48, 48), (120, 40, 80)).save(path)
        event = {
            "event_id": f"event-{index}", "frame_id": f"frame-{index}",
            "session_id": f"session-{index}", "observed_at": f"2026-10-0{index + 1}",
            "sensor_assets": {"color_preview": {
                "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
            "detections": [{"box": [0, 0, 1, 1], "label": "object", "track_id": f"track-{index}"}],
        }
        connection.execute("insert into observations values (?)", (json.dumps(event),))
    connection.commit()
    connection.close()
    output = tmp_path / "dataset"
    exporter.export_archive(archive, output)
    dataset = json.loads((output / "dataset.json").read_text())
    assert dataset["eligible_unique_crops"] == 1
    row, = dataset["entries"]
    assert row["split"] == "test"
    assert row["session_id"] == "session-2" and row["frame_id"] == "frame-2"
    assert row["scene_file"] == "scenes/frame-2.jpg"
    assert row["source_observations"] == ["event-0", "event-1", "event-2"]
    assert len(row["source_occurrences"]) == 3
