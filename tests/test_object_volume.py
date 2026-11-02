from types import SimpleNamespace

import numpy as np
import pytest

from rgb.object_volume import cloud_estimate, estimate_objects, volume_label


def frame(depth=None):
    shape = (288, 320)
    plane = {'intrinsics': {'width': 320, 'height': 288, 'fx': 300., 'fy': 300.,
                            'cx': 160., 'cy': 144.}, 'distortion': {'model': 'NONE'}}
    metadata = {'frame_id': 'frame', 'session_id': 'session', 'calibration_sha256': 'a'*64,
                'depth_scale_mm': 1., 'profiles': {'depth': {'width': 320, 'height': 288}},
                'calibration': {'planes': {'color': plane, 'depth': {
                    **plane, 'extrinsics_to_color': {'rotation': np.eye(3).ravel().tolist(),
                                                    'translation_mm': [0, 0, 0]}}}}}
    return SimpleNamespace(id='frame', metadata=metadata,
                           color_rgb=np.zeros((*shape, 3), np.uint8),
                           depth_native=depth, depth_registered=depth)


def detection(label='backpack'):
    return {'box': [0, 0, 1, 1], 'label': label}


def cylinder():
    angle, height = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 101), np.linspace(0, .1, 51))
    return np.column_stack([(.04*np.sin(angle)).ravel(), height.ravel(),
                            (1-.04*np.cos(angle)).ravel()])


def test_shape_closure_requires_axis_and_reports_mask_truncation():
    points = cylinder()
    vertical = {'normal': [0, 1, 0], 'assumption': 'horizontal reference'}
    result = cloud_estimate(points, 'paper cup', vertical)
    assert result['method'] == 'upright_cylinder'
    assert volume_label(result).endswith('(cyl)')
    assert result['estimate_m3'] == pytest.approx(np.pi*.04**2*.1, rel=.06)
    assert result['physical_accuracy'] == 'unverified'
    assert result['dimensions_m'][0] == pytest.approx(.08, rel=.01)
    missing = cloud_estimate(points, 'cup')
    assert missing['method'] == 'trimmed_visible_point_box'
    assert '(cyl)' not in volume_label(missing)
    assert missing['cylinder']['status'] == 'no_vertical_reference'
    clipped = cloud_estimate(points, 'cup', vertical, truncated=True)
    assert clipped['cylinder']['status'] == 'truncated_object'
    assert clipped['method'] != 'upright_cylinder'
    mixed = cloud_estimate(np.vstack([points, points+[0, 0, 1]]), 'bottle', vertical)
    assert mixed['cylinder']['status'] == 'inconsistent_depth_span'


def test_planar_or_missing_geometry_never_becomes_zero_full_object_volume():
    xx, yy = np.meshgrid(np.linspace(0, .2, 30), np.linspace(0, .3, 40))
    points = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    result = cloud_estimate(points, 'whiteboard')
    assert result['status'] == 'visible_surface_only'
    assert result['estimate_m3'] is None
    assert volume_label(result) == 'volume unavailable'
    assert cloud_estimate(points[:5], 'bottle')['estimate_m3'] is None


@pytest.mark.parametrize('value,status', [(0, 'insufficient_depth'),
                                         (65535, 'insufficient_depth'),
                                         (6000, 'outside_operating_range')])
def test_invalid_and_out_of_mode_depth_cannot_create_volume(value, status):
    f = frame(np.full((288, 320), value, np.uint16))
    readings, summary = estimate_objects(f, [detection()], [np.ones((288, 320), bool)])
    assert readings[0]['status'] == status
    assert readings[0]['estimate_m3'] is None
    assert readings[0]['native_samples'] == 0
    assert readings[0]['source_frame_id'] == f.id
    assert summary['operating_range_m'] == [.5, 5.46]


def test_no_rgb_only_box_fallback_and_invalid_calibration_isolated():
    f = frame(np.full((288, 320), 1200, np.uint16))
    readings, _ = estimate_objects(f, [detection(), detection()], [None, np.zeros((288, 320), bool)])
    assert [r['status'] for r in readings] == ['no_instance_mask', 'empty_instance_mask']
    assert all(r['estimate_m3'] is None for r in readings)
    f.metadata['calibration'] = {}
    readings, summary = estimate_objects(f, [detection()], [np.ones((288, 320), bool)])
    assert readings[0]['status'] == 'invalid_calibration_or_depth'
    assert 'geometry_error' in summary
    readings, _ = estimate_objects(frame(), [detection()], [None])
    assert readings[0]['status'] == 'no_native_depth'
    with pytest.raises(ValueError, match='correspond'):
        estimate_objects(f, [detection()], [])


def test_native_mask_selection_avoids_box_background_and_preserves_detection_order():
    yy, xx = np.mgrid[:288, :320]
    depth = np.full((288, 320), 6000, np.uint16)
    mask = (yy >= 60) & (yy < 140) & (xx >= 60) & (xx < 140)
    depth[mask] = 1000+((xx[mask]-60)*2).astype(np.uint16)
    f = frame(depth)
    readings, _ = estimate_objects(f, [detection('far chair'), detection()], [~mask, mask])
    assert readings[0]['status'] == 'outside_operating_range'
    assert readings[1]['raw_native_samples'] == int(mask.sum())
    assert readings[1]['outside_operating_range_samples'] == 0
    assert readings[1]['source_box'] == [0, 0, 1, 1]
    assert readings[1]['mask_sensitivity']['native_samples'] < readings[1]['native_samples']
    assert readings[1]['registered_valid_fraction'] == 1.


def test_unsupported_profile_does_not_guess_an_operating_range():
    f = frame(np.ones((288, 320), np.uint16))
    f.depth_native = np.ones((20, 30), np.uint16)
    f.metadata['profiles']['depth'].update(width=30, height=20)
    readings, _ = estimate_objects(f, [detection()], [np.ones((288, 320), bool)])
    assert readings[0]['status'] == 'unsupported_depth_profile'
