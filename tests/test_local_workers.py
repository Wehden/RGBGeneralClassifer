"""Exercise orchestration around detector/runtime/camera dependency boundaries."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rgb.camera import CameraUnavailable, make_frame
from test_local import local


@pytest.mark.parametrize('backend', ['orbbec', 'realsense', 'v4l2'])
def test_detector_loop_publishes_paired_results_and_only_measures_depth_sources(local, monkeypatch, backend):
    store, camera, frame, result, runtime, _ = local
    camera.backend = backend
    depth = backend != 'v4l2'
    if not depth:
        camera.backend = 'v4l2'
        frame = make_frame(frame.color_rgb.copy(), None, None, None,
                           {**frame.metadata, 'frame_id': str(uuid4())})
        camera.publish(frame)
    store.update_settings({'detector_threshold': .35})
    calls = []
    model = SimpleNamespace(
        manifest={'model': 'synthetic-detector'}, sha256='a' * 64,
        config={'runtime': 'synthetic'}, instance_masks=[None], mask_error=None)
    def detect(image, *, with_masks):
        calls.append((image.shape, with_masks, model.threshold))
        runtime.stop_event.set()
        return [deepcopy(result['detections'][0])], 12.
    model.detect = detect
    monkeypatch.setattr('board.yoloe.Detector', lambda *args, **kwargs: model)
    runtime.live_readings['old-session'] = {}
    runtime.material_cache['old-session'] = {}
    runtime._detector()
    published_frame, published = camera.latest_detection
    assert published_frame.id == frame.id
    assert calls == [(frame.color_rgb.shape, depth, .35)]
    assert camera.detector_status['state'] == 'running'
    assert 'old-session' not in runtime.material_cache
    assert 'old-session' not in runtime.live_readings
    assert runtime.material_pending[0].id == frame.id
    assert runtime.material_pending[2]['threshold'] == .35
    assert published['detections'][0]['track_id']
    assert ('volume' in published['detections'][0]) is depth
    assert ('metrology' in published) is depth
    assert ('volume_basis' in published['detector']['config']) is depth


def test_missing_detector_artifact_reports_unavailable_without_starting_worker(local, monkeypatch):
    _, camera, _, _, runtime, _ = local
    def broken(*args, **kwargs):
        raise ValueError('checksum mismatch')
    monkeypatch.setattr('board.yoloe.Detector', broken)
    runtime._detector()
    assert camera.detector_status['state'] == 'unavailable'
    assert 'checksum mismatch' in camera.detector_status['error']
    assert runtime.material_pending is None


def test_camera_outage_clears_live_opinions_and_exits_on_shutdown(local, monkeypatch):
    _, camera, _, _, runtime, _ = local
    model = SimpleNamespace(manifest={'model': 'synthetic'}, sha256='a' * 64, config={})
    monkeypatch.setattr('board.yoloe.Detector', lambda *args, **kwargs: model)
    def absent(previous):
        raise CameraUnavailable('synthetic disconnect')
    monkeypatch.setattr(camera, 'wait', absent)
    monkeypatch.setattr(runtime.stop_event, 'wait', lambda seconds: runtime.stop_event.set() or True)
    runtime.live_readings['old'] = {}
    runtime._detector()
    assert not runtime.live_readings


def test_missing_material_artifact_reports_unavailable_without_processing_pending_work(local, monkeypatch):
    _, _, _, _, runtime, _ = local
    def broken(*args, **kwargs):
        raise ValueError('checksum mismatch')
    monkeypatch.setattr('rgb.materials.MaterialClassifier', broken)
    runtime._materials()
    assert runtime.material_status['state'] == 'unavailable'
    assert 'checksum mismatch' in runtime.material_status['error']
