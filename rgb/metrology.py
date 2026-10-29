"""Experimental quantitative depth geometry, separate from the preview cloud."""
import math

import cv2
import numpy as np


class UnsupportedPlane(ValueError):
    """Valid points do not provide enough coherent support for a plane."""


def camera_matrix(plane, shape):
    intrinsic = plane['intrinsics']
    h, w = shape
    if (intrinsic['width'], intrinsic['height']) != (w, h):
        raise ValueError('Depth calibration dimensions mismatch')
    fx, fy, cx, cy = [float(intrinsic[name]) for name in ['fx', 'fy', 'cx', 'cy']]
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid camera intrinsics')
    model = str(plane.get('distortion', {}).get('model', 'NONE')).upper()
    if 'BROWN_CONRADY' not in model and model not in {'NONE', 'OBCAMERADISTORTIONMODEL.NONE'}:
        raise ValueError('Unsupported distortion model')
    coefficients = np.array([float(plane.get('distortion', {}).get(key, 0))
                             for key in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']])
    if not np.isfinite(coefficients).all():
        raise ValueError('Invalid distortion coefficients')
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], np.float64), coefficients


def backproject(depth, plane, scale_mm, min_range=.5, max_range=3.86):
    """Return independent native depth points and corresponding native pixels."""
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('Expected native uint16 depth')
    if (not np.isfinite([scale_mm, min_range, max_range]).all()
            or scale_mm <= 0 or not 0 <= min_range < max_range):
        raise ValueError('Invalid depth scale or range')
    matrix, coefficients = camera_matrix(plane, depth.shape)
    z = depth.astype(np.float64) * scale_mm / 1000
    yy, xx = np.nonzero((z > 0) & (z >= min_range) & (z <= max_range))
    pixels = np.stack([xx, yy], axis=1).astype(np.float64)
    if not len(pixels):
        return np.empty((0, 3)), pixels
    rays = cv2.undistortPoints(
        pixels[:, None], matrix, coefficients,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10))[:, 0]
    depths = z[yy, xx]
    points = np.column_stack([rays * depths[:, None], depths])
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite backprojected geometry')
    return points, pixels


def native_mask_points(depth, metadata, masks, *, min_range=.5, max_range=3.86):
    """Select native-camera points with RGB masks, retaining factory transform diagnostics."""
    planes = metadata['calibration']['planes']
    points, pixels = backproject(depth, planes['depth'], metadata['depth_scale_mm'],
                                min_range=min_range, max_range=max_range)
    color_shape = (planes['color']['intrinsics']['height'], planes['color']['intrinsics']['width'])
    if any(mask.shape != color_shape or mask.dtype != np.bool_ for mask in masks):
        raise ValueError('RGB mask dimensions or dtype mismatch')
    matrix, coefficients = camera_matrix(planes['color'], color_shape)
    extrinsic = planes['depth']['extrinsics_to_color']
    rotation = np.asarray(extrinsic['rotation'], np.float64).reshape(3, 3)
    translation = np.asarray(extrinsic['translation_mm'], np.float64).reshape(3) / 1000
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Invalid depth-to-color transform')
    diagnostics = {
        'rotation_orthogonality_error': float(np.max(np.abs(rotation.T @ rotation-np.eye(3)))),
        'rotation_determinant': float(np.linalg.det(rotation)),
        'coordinate_frame': 'native_depth_camera', 'units': 'metres',
        'independent_valid_depth_samples': len(points),
        'selection_range_m': [min_range, max_range],
    }
    if diagnostics['rotation_orthogonality_error'] > .001:
        diagnostics['calibration_warning'] = 'Factory rotation is not orthonormal; RGB mask projection requires qualification'
    if not len(points):
        return [points.copy() for _ in masks], diagnostics
    color_points = points @ rotation.T + translation
    projected = cv2.projectPoints(color_points, np.zeros(3), np.zeros(3), matrix, coefficients)[0][:, 0]
    valid = (np.isfinite(projected).all(axis=1) & (color_points[:, 2] > 0)
             & (projected[:, 0] >= -.5) & (projected[:, 0] < color_shape[1]-.5)
             & (projected[:, 1] >= -.5) & (projected[:, 1] < color_shape[0]-.5))
    indices = np.flatnonzero(valid)
    xy = np.floor(projected[valid]+.5).astype(np.int64)
    linear = xy[:, 1]*color_shape[1]+xy[:, 0]
    front = np.full(color_shape[0]*color_shape[1], np.inf)
    np.minimum.at(front, linear, color_points[indices, 2])
    visible = color_points[indices, 2] <= front[linear]+.01
    selections = []
    for mask in masks:
        chosen = visible & mask[xy[:, 1], xy[:, 0]]
        selections.append(points[indices[chosen]])
    diagnostics['selection'] = 'factory depth-to-color projection; nearest RGB pixel; 10 mm occlusion tolerance'
    # Measurements stay in native coordinates: the factory transform only selects points.
    return selections, diagnostics


def fit_plane(points, threshold_m=.008, trials=100, seed=20261028):
    """Robust candidate plane; caller must establish that it is the support surface."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        raise ValueError('At least 30 finite points are required')
    if not math.isfinite(threshold_m) or threshold_m <= 0 or trials < 1:
        raise ValueError('Invalid plane-fit parameters')
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), 8000), replace=False)]
    best = np.zeros(len(sample), bool)
    for _ in range(trials):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        inliers = np.abs(sample @ normal-a @ normal) <= threshold_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 30:
        raise UnsupportedPlane('No supported plane found')
    center = sample[best].mean(axis=0)
    _, _, vectors = np.linalg.svd(sample[best]-center, full_matrices=False)
    normal = vectors[-1]
    offset = -float(center @ normal)
    if offset < 0:
        normal = -normal
        offset = -offset
    residuals = np.abs(points @ normal+offset)
    inliers = residuals <= threshold_m
    return {'normal': normal.tolist(), 'offset_m': offset,
            'inlier_fraction': float(inliers.mean()),
            'inlier_residual_p95_m': float(np.percentile(residuals[inliers], 95)),
            'support_surface_identity': 'unverified'}


def heightfield(points, normal, offset_m, cell_size_m=.005, expected_cells=None):
    """Observed integral only; missing footprint cells are never silently filled."""
    points = np.asarray(points, np.float64)
    normal = np.asarray(normal, np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or normal.shape != (3,)
            or not np.isfinite(normal).all() or not math.isfinite(offset_m)
            or not math.isfinite(cell_size_m) or cell_size_m <= 0
            or not np.isclose(np.linalg.norm(normal), 1)):
        raise ValueError('Invalid height-field geometry')
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis);u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    heights = points @ normal+offset_m
    valid = heights > 0
    coordinates = np.column_stack([points[valid] @ u, points[valid] @ v])
    cells = np.floor(coordinates/cell_size_m+1e-10).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    values = [float(np.median(heights[valid][inverse == i])) for i in range(len(unique))]
    observed = {tuple(cell): height for cell, height in zip(unique.tolist(), values)}
    if expected_cells is None:
        integral = sum(values)*cell_size_m**2
        coverage = None
        status = 'footprint_unverified'
    else:
        expected = {tuple(cell) for cell in expected_cells}
        if not expected:
            raise ValueError('Expected footprint must not be empty')
        integral = sum(observed.get(cell, 0) for cell in expected)*cell_size_m**2
        coverage = len(expected.intersection(observed))/len(expected)
        status = 'complete_under_heightfield_assumption' if coverage == 1 else 'partial'
    return {'observed_integral_m3': integral, 'cell_size_m': cell_size_m,
            'observed_cells': len(unique), 'footprint_coverage': coverage, 'status': status,
            'filled_cells': 0, 'assumption': 'solid fills between observed top surface and support plane'}
