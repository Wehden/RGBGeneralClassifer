import numpy as np
import pytest

from rgb.capture_metrology import observed_range, upright_cylinder, visible_envelope
from rgb.metrology import native_mask_points


def front_of_cylinder():
    angle, height = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 101),
                                np.linspace(0, .1, 51))
    return np.column_stack([(.04*np.sin(angle)).ravel(), height.ravel(),
                            (1-.04*np.cos(angle)).ravel()])


def test_full_observed_range_preserves_far_samples_and_excludes_invalid_sentinels():
    depth = np.array([[0, 661, 3860, 7778, 65535]], dtype=np.uint16)
    result = observed_range(depth, 1)
    assert result['min_m'] == pytest.approx(.661)
    assert result['max_m'] == pytest.approx(7.778)
    assert result['samples'] == 3 and result['saturated_samples'] == 1
    assert result['nonzero_unsaturated_fraction'] == pytest.approx(.6)
    assert 'min_m' not in observed_range(np.zeros((2, 2), np.uint16), 1)
    with pytest.raises(ValueError):
        observed_range(depth, float('nan'))


def test_native_mask_selection_can_explicitly_include_far_depth_without_changing_default():
    plane = {'intrinsics': {'width': 2, 'height': 1, 'fx': 10., 'fy': 10., 'cx': 0., 'cy': 0.},
             'distortion': {'model': 'NONE'}}
    metadata = {'depth_scale_mm': 1., 'calibration': {'planes': {
        'color': plane, 'depth': {**plane, 'extrinsics_to_color': {
            'rotation': np.eye(3).ravel().tolist(), 'translation_mm': [0, 0, 0]}}}}}
    depth = np.array([[1000, 5000]], dtype=np.uint16)
    mask = np.ones((1, 2), bool)
    assert len(native_mask_points(depth, metadata, [mask])[0][0]) == 1
    clouds, diagnostic = native_mask_points(depth, metadata, [mask], min_range=0, max_range=8)
    assert len(clouds[0]) == 2 and clouds[0][1, 2] == 5
    assert diagnostic['selection_range_m'] == [0, 8]


def test_cylinder_infers_external_closure_from_front_surface_and_is_rotation_invariant():
    points = front_of_cylinder()
    result = upright_cylinder(points, [0, 1, 0], trim_percent=0)
    assert result['external_envelope_l'] == pytest.approx(np.pi*.04**2*.1*1000)
    assert result['height_m'] == pytest.approx(.1)
    assert result['diameter_m'] == pytest.approx(.08)
    assert result['physical_accuracy'] == 'unverified'
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated = upright_cylinder(points @ rotation.T, rotation @ [0, 1, 0], trim_percent=0)
    assert rotated['external_envelope_l'] == pytest.approx(result['external_envelope_l'])


def test_multiple_depth_layers_and_missing_depth_do_not_produce_cylinder_volume():
    points = front_of_cylinder()
    mixed = np.vstack([points, points+[0, 0, 1]])
    assert upright_cylinder(mixed, [0, 1, 0])['status'] == 'inconsistent_depth_span'
    assert upright_cylinder(points[:20], [0, 1, 0])['status'] == 'insufficient_depth'
    with pytest.raises(ValueError):
        upright_cylinder(points, [0, 0, 0])


def test_flat_surface_envelope_is_explicitly_partial_and_points_must_be_finite():
    xx, yy = np.meshgrid(np.linspace(0, .2, 30), np.linspace(0, .3, 40))
    points = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    result = visible_envelope(points)
    assert result['status'] == 'visible_surface_only'
    assert result['envelope_l'] == pytest.approx(0)
    with pytest.raises(ValueError):
        visible_envelope(np.array([[0, 0, np.nan]]))
