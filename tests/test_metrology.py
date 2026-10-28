"""Analytic references for experimental masks and quantitative depth geometry."""
import cv2
import numpy as np
import pytest

from rgb.instance_masks import instances
from rgb.metrology import backproject, fit_plane, heightfield, native_mask_points


def plane(width=13, height=11):
    return {'intrinsics':{'width':width,'height':height,'fx':25.,'fy':24.,'cx':6.,'cy':5.},
            'distortion':{'model':'BROWN_CONRADY_K6','k1':.1,'k2':-.03,'p1':.001,'p2':-.002}}


def box_surface():
    # Top of a 200 x 300 x 100 mm cuboid, observed on a 10 mm metric grid.
    xx,yy = np.meshgrid((np.arange(20)+.5)*.01,(np.arange(30)+.5)*.01)
    points = np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,.9)])
    footprint = [(-y-1,-x-1) for y in range(30) for x in range(20)]
    return points, footprint


def test_depth_distortion_round_trip_and_mm_scale():
    depth = np.full((11,13),1001,np.uint16)
    reference = plane()
    points,pixels = backproject(depth,reference,1.)
    matrix = np.array([[25.,0,6.],[0,24.,5.],[0,0,1.]])
    coefficients = np.array([.1,-.03,.001,-.002,0,0,0,0])
    projected = cv2.projectPoints(points,np.zeros(3),np.zeros(3),matrix,coefficients)[0][:,0]
    assert np.max(np.abs(projected-pixels)) < 1e-6
    assert np.allclose(points[:,2],1.001)


def test_native_depth_rejects_invalid_samples_and_calibration():
    depth = np.array([[0,499,500,3860,3861]],np.uint16)
    reference = plane(5,1)
    points,_ = backproject(depth,reference,1.)
    assert points[:,2] == pytest.approx([.5,3.86])
    with pytest.raises(ValueError):backproject(depth,reference,0)
    with pytest.raises(ValueError):backproject(depth,plane(),1)
    reference['intrinsics']['fx'] = float('nan')
    with pytest.raises(ValueError):backproject(depth,reference,1)


def test_depth_to_rgb_uses_extrinsic_mm_and_retains_native_geometry():
    calibration = plane(12,10)
    calibration['intrinsics'].update(fx=100.,fy=100.,cx=0.,cy=0.)
    calibration['distortion'] = {'model':'NONE'}
    depth_plane = {**calibration,'extrinsics_to_color':{'rotation':np.eye(3).ravel().tolist(),
                                                       'translation_mm':[10,0,0]}}
    metadata = {'depth_scale_mm':1.,'calibration':{'planes':{'depth':depth_plane,'color':calibration}}}
    depth = np.zeros((10,12),np.uint16);depth[4,2:4] = 1000
    mask = np.zeros((10,12),bool);mask[4,3] = True
    selections,diagnostics = native_mask_points(depth,metadata,[mask])
    assert selections[0].shape == (1,3)
    assert selections[0][0] == pytest.approx([.02,.04,1.])
    assert diagnostics['coordinate_frame'] == 'native_depth_camera'
    assert diagnostics['rotation_orthogonality_error'] == 0


def test_heightfield_recovers_known_cuboid_volume_and_does_not_fill_missing_cells():
    points,footprint = box_surface()
    result = heightfield(points,[0,0,-1],1.,.01,footprint)
    assert result['observed_integral_m3'] == pytest.approx(.2*.3*.1)
    assert result['footprint_coverage'] == 1
    assert result['status'] == 'complete_under_heightfield_assumption'
    partial = heightfield(points[60:],[0,0,-1],1.,.01,footprint)
    assert partial['observed_integral_m3'] == pytest.approx(.006*.9)
    assert partial['footprint_coverage'] == .9
    assert partial['status'] == 'partial'
    assert partial['filled_cells'] == 0
    assert heightfield(points,[0,0,-1],1.)['status'] == 'footprint_unverified'


def test_sphere_demonstrates_heightfield_assumption_overestimates_external_volume():
    radius = .1;step = .002
    centers = np.arange(-radius+step/2,radius,step)
    xx,yy = np.meshgrid(centers,centers)
    valid = xx**2+yy**2 < radius**2
    top = radius+np.sqrt(radius**2-xx[valid]**2-yy[valid]**2)
    points = np.column_stack([xx[valid],yy[valid],1-top])
    result = heightfield(points,[0,0,-1],1.,step)
    actual_volume = 4/3*np.pi*radius**3
    # Filling below the visible hemisphere adds 25% to a complete sphere.
    assert result['observed_integral_m3']/actual_volume == pytest.approx(1.25,abs=.01)
    assert result['status'] == 'footprint_unverified'


def test_candidate_plane_fit_with_noise_and_outliers_preserves_identity_uncertainty():
    rng = np.random.default_rng(27)
    xy = rng.uniform(-.4,.4,(1000,2))
    surface = np.column_stack([xy,1+rng.normal(0,.001,1000)])
    outliers = rng.uniform([-.4,-.4,.5],[.4,.4,1.4],(100,3))
    result = fit_plane(np.vstack([surface,outliers]),threshold_m=.005)
    assert result['normal'] == pytest.approx([0,0,-1],abs=.001)
    assert result['offset_m'] == pytest.approx(1,abs=.001)
    assert result['inlier_fraction'] > .90
    assert result['support_surface_identity'] == 'unverified'


def test_mask_padding_is_removed_and_boxes_crop_in_source_coordinates():
    output = np.zeros((1,1,38),np.float32)
    output[0,0,:6] = [0,40,320,280,.9,0]
    output[0,0,6] = 1
    prototypes = np.zeros((1,32,80,80),np.float32)
    prototypes[0,0] = 1
    prototypes[0,0,:10] = -10;prototypes[0,0,70:] = -10
    result = instances(output,prototypes,['box'],(480,640,3),(.5,0,40))
    assert len(result) == 1
    assert result[0]['mask'].shape == (480,640)
    assert result[0]['mask'].all()
    output[0,0,0] = 160
    result = instances(output,prototypes,['box'],(480,640,3),(.5,0,40))
    assert not result[0]['mask'][:,:320].any()
    assert result[0]['mask'][:,320:].all()


def test_masks_reject_unexpected_contract_and_nonfinite_prototypes():
    with pytest.raises(ValueError):
        instances(np.zeros((1,1,6)),np.zeros((1,32,80,80)),['box'],(320,320,3),(1,0,0))
    output = np.zeros((1,1,38),np.float32)
    prototypes = np.zeros((1,32,80,80),np.float32);prototypes[0,0,0,0] = float('nan')
    with pytest.raises(ValueError):
        instances(output,prototypes,['box'],(320,320,3),(1,0,0))
