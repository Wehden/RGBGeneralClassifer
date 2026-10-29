import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rgb.metrology import UnsupportedPlane, fit_plane

spec = importlib.util.spec_from_file_location(
    'capture_volume_runner', Path(__file__).resolve().parents[1]/'scripts/estimate-capture-volume.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    plane = {'intrinsics': {'width': 2, 'height': 2, 'fx': 10., 'fy': 10., 'cx': 0., 'cy': 0.},
             'distortion': {'model': 'NONE'}}
    paths = []
    for index in range(2):
        metadata = {'frame_id': f'frame-{index}', 'received_at': '2026-10-29T15:00:00+00:00',
                    'calibration_sha256': 'b'*64, 'depth_scale_mm': 1.,
                    'registered_depth_scale_mm': 1., 'profiles': {'depth': {'width': 2, 'height': 2}},
                    'calibration': {'planes': {'depth': plane}}}
        path = tmp_path/f'frame-{index}.npz'
        np.savez_compressed(path, color_rgb=np.zeros((16, 16, 3), np.uint8),
                            depth_native=np.full((2, 2), 1000, np.uint16),
                            depth_registered=np.full((16, 16), 1000, np.uint16),
                            metadata_json_utf8=np.frombuffer(json.dumps(metadata).encode(), np.uint8))
        paths.append(path)
    model = SimpleNamespace(size=32, input_name='images', model='test-detector', sha256='a'*64,
                            threshold=.45, labels=['ceiling', 'cup'],
                            session=SimpleNamespace(run=Mock(return_value=[None, None])))
    # Valid finite points spread through a volume, with no supported plane.
    noise = np.random.default_rng(37).uniform([-1, -1, .5], [1, 1, 2.5], (120, 3))
    angle, height = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 31), np.linspace(0, .1, 11))
    cup = np.column_stack([(.04*np.sin(angle)).ravel(), height.ravel(),
                           (1-.04*np.cos(angle)).ravel()])
    monkeypatch.setattr(runner, 'Detector', lambda *args, **kwargs: model)

    def configure(labels, clouds):
        detections = [{'label': label, 'class_id': i, 'confidence': .9, 'box': [0, 0, 1, 1],
                       'mask': np.ones((16, 16), bool)} for i, label in enumerate(labels)]
        monkeypatch.setattr(runner, 'instances', lambda *args: detections)
        monkeypatch.setattr(runner, 'native_mask_points', lambda *args, **kwargs: (clouds, {}))

    return SimpleNamespace(paths=paths, model=model, noise=noise, cup=cup, configure=configure)


def test_unsupported_ceiling_preserves_all_batch_receipts_without_cylinder_estimates(scene, tmp_path, monkeypatch):
    with pytest.raises(UnsupportedPlane, match='No supported plane found'):
        fit_plane(scene.noise, threshold_m=.01)
    scene.configure(['ceiling', 'cup'], [scene.noise, scene.cup])
    output = tmp_path/'results'
    monkeypatch.setattr(runner.sys, 'argv', ['estimate-capture-volume', '--frames',
                        *(str(p) for p in scene.paths), '--model-dir', str(tmp_path),
                        '--output-dir', str(output)])
    runner.main()
    assert json.loads((output/'summary.json').read_text())['frames'] == 2
    for path in scene.paths:
        receipt = json.loads((output/(path.stem+'-measurements.json')).read_text())
        assert receipt['vertical_reference'] is None
        assert receipt['vertical_reference_rejections'] == [
            {'detection_index': 0, 'samples': 120, 'reason': 'no_supported_plane'}]
        assert receipt['range']['min_m'] == receipt['range']['max_m'] == 1
        cup = receipt['objects'][1]
        assert cup['visible_envelope']['samples'] == len(scene.cup)
        assert cup['volume_status'] == 'no_vertical_reference' and 'cylinder' not in cup
        assert (output/(path.stem+'-objects.jpg')).is_file()


def test_rejected_ceiling_does_not_prevent_a_later_supported_candidate(scene, tmp_path):
    xx, zz = np.meshgrid(np.linspace(-.3, .3, 15), np.linspace(.8, 1.5, 15))
    plane = np.column_stack([xx.ravel(), np.full(xx.size, -1.), zz.ravel()])
    scene.configure(['ceiling', 'ceiling', 'cup'], [scene.noise, plane, scene.cup])
    receipt = runner.analyse(scene.paths[0], scene.model, tmp_path)
    assert receipt['vertical_reference']['samples'] == len(plane)
    assert receipt['vertical_reference']['inlier_fraction'] == pytest.approx(1)
    assert receipt['vertical_reference_rejections'][0]['detection_index'] == 0
    assert receipt['objects'][2]['volume_status'] == 'conditional_estimate'


def test_invalid_plane_inputs_are_not_silently_treated_as_unsupported(scene, tmp_path, monkeypatch):
    scene.configure(['ceiling', 'cup'], [scene.noise, scene.cup])
    monkeypatch.setattr(runner, 'fit_plane', lambda *args, **kwargs: fit_plane(np.full((120, 3), np.nan)))
    with pytest.raises(ValueError, match='finite points') as error:
        runner.analyse(scene.paths[0], scene.model, tmp_path)
    assert not isinstance(error.value, UnsupportedPlane)
