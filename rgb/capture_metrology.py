"""Observed range and explicit single-view shape assumptions; no accuracy claim."""
import math

import numpy as np


def observed_range(depth, scale_mm):
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError('Expected native uint16 depth')
    if not math.isfinite(scale_mm) or scale_mm <= 0:
        raise ValueError('Invalid depth scale')
    valid = (depth > 0) & (depth < np.iinfo(np.uint16).max)
    values = depth[valid].astype(np.float64) * scale_mm / 1000
    result = {'samples': len(values), 'nonzero_unsaturated_fraction': float(valid.mean()),
              'missing_fraction': float((depth == 0).mean()),
              'saturated_samples': int((depth == np.iinfo(np.uint16).max).sum()),
              'semantics': 'observed axial depth, not device rated range or verified accuracy'}
    if len(values):
        result.update(min_m=float(values.min()), max_m=float(values.max()),
                      p01_m=float(np.percentile(values, 1)), p99_m=float(np.percentile(values, 99)),
                      median_m=float(np.median(values)))
    return result


def visible_envelope(points, trim_percent=2):
    """PCA box enclosing a trimmed visible cloud, not a complete object volume."""
    points = _points(points, trim_percent)
    if len(points) < 30:
        return {'status': 'insufficient_depth', 'samples': len(points)}
    low, high = np.percentile(points, [trim_percent, 100-trim_percent], axis=0)
    retained = points[np.all((points >= low) & (points <= high), axis=1)]
    if len(retained) < 30:
        return {'status': 'insufficient_depth', 'samples': len(retained)}
    center = np.median(retained, axis=0)
    _, _, axes = np.linalg.svd(retained-center, full_matrices=False)
    limits = np.percentile((retained-center) @ axes.T, [trim_percent, 100-trim_percent], axis=0)
    dimensions = limits[1]-limits[0]
    return {'status': 'visible_surface_only' if dimensions.min() < .01 or dimensions.min() < .05*dimensions.max()
            else 'partial_visible_envelope',
            'samples': len(points), 'retained_samples': len(retained),
            'trim_percent': trim_percent, 'dimensions_m': dimensions.tolist(),
            'envelope_l': float(np.prod(dimensions)*1000),
            'axes_native_camera': axes.tolist(),
            'semantics': 'trimmed visible-point box proxy; neither full object volume nor an accuracy bound'}


def upright_cylinder(points, vertical_axis, trim_percent=2):
    """Infer unseen cylindrical closure using observed height and silhouette width."""
    points = _points(points, trim_percent)
    axis = np.array(vertical_axis, dtype=np.float64, copy=True)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError('Expected a finite nonzero vertical axis')
    axis /= np.linalg.norm(axis)
    if len(points) < 100:
        return {'status': 'insufficient_depth', 'samples': len(points)}
    center = np.median(points, axis=0)
    view = center-axis*(center @ axis)
    if np.linalg.norm(view) < 1e-6:
        return {'status': 'axis_parallel_to_view'}
    width_axis = np.cross(axis, view)
    width_axis /= np.linalg.norm(width_axis)
    height = float(np.diff(np.percentile(points @ axis, [trim_percent, 100-trim_percent]))[0])
    diameter = float(np.diff(np.percentile(points @ width_axis, [trim_percent, 100-trim_percent]))[0])
    depth_span = float(np.diff(np.percentile(points[:, 2], [trim_percent, 100-trim_percent]))[0])
    if height < .01 or diameter < .01:
        return {'status': 'insufficient_extent'}
    # A long ray-depth span relative to silhouette width suggests mask leakage,
    # multiple surfaces, or a shape incompatible with this closure assumption.
    if depth_span > 3*diameter:
        return {'status': 'inconsistent_depth_span', 'depth_span_m': depth_span,
                'diameter_m': diameter}
    return {'status': 'conditional_estimate', 'height_m': height, 'diameter_m': diameter,
            'external_envelope_l': math.pi/4*diameter**2*height*1000,
            'trim_percent': trim_percent, 'samples': len(points),
            'assumptions': ['upright relative to supplied axis', 'circular cylinder',
                            'visible mask spans object width and height',
                            'depth belongs to the detected object'],
            'semantics': 'solid external cylinder envelope, not vessel capacity or measured composition',
            'physical_accuracy': 'unverified'}


def _points(points, trim_percent):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points')
    if not math.isfinite(trim_percent) or not 0 <= trim_percent < 25:
        raise ValueError('Invalid trimming percentile')
    return points
