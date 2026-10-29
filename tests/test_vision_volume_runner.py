"""Dataset coverage is independent of the latency sampling budget."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'vision_volume_runner', Path(__file__).resolve().parents[1]/'scripts/benchmark-vision-volume.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('samples', [1, 3, 5])
def test_detector_receipt_covers_all_frames_with_independent_latency_samples(tmp_path, monkeypatch, samples):
    frames = []
    for i in range(3):
        path = tmp_path/f'frame-{i}.npz'
        np.savez(path, color_rgb=np.full((4, 4, 3), i, dtype=np.uint8),
                 depth_native=np.zeros((4, 4), dtype=np.uint16),
                 depth_registered=np.zeros((4, 4), dtype=np.uint16),
                 metadata_json_utf8=np.frombuffer(json.dumps({'frame_id':str(i)}).encode(), dtype=np.uint8))
        frames.append(path)
    seen = []
    def infer(model, image):
        seen.append(int(image[0, 0, 0]))
        return [], None, len(seen)
    monkeypatch.setattr(runner, 'Detector', lambda *a, **kw: SimpleNamespace(config={}))
    monkeypatch.setattr(runner, 'infer', infer)
    monkeypatch.setattr(runner, 'mask_reference', lambda *a: {'passed':True})
    measured = []
    def native(depth, metadata, masks):
        measured.append(metadata['frame_id'])
        return [], {}
    monkeypatch.setattr(runner, 'native_mask_points', native)
    result = runner.detector_worker(tmp_path/'detector', frames, tmp_path/'out', samples, 1)
    assert result['frames'] == len(result['scene_readings']) == 3
    assert [row['source_file'] for row in result['scene_readings']] == list(map(str, frames))
    assert measured == ['0', '1', '2']
    assert len(result['mask_reference_parity']) == 3
    assert len(seen) == 5 + max(samples, 3)  # Five warmups; each dataset frame still inferred.
    assert result['samples'] == samples
    assert result['inference_p50_ms'] == np.median(range(6, 6 + samples))
