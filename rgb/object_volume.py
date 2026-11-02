"""Per-instance single-view depth estimates, with explicit closure assumptions."""
import time

import cv2
import numpy as np

from rgb.capture_metrology import upright_cylinder, visible_envelope
from rgb.metrology import UnsupportedPlane, fit_plane, native_mask_points


# Femto Bolt profile ranges; dimensions identify the active native depth mode.
# https://www.orbbec.com/documentation/femto-bolt-hardware-specifications/
OPERATING_RANGES = {
    (640, 576): (.5, 3.86),
    (320, 288): (.5, 5.46),
    (512, 512): (.25, 2.88),
    (1024, 1024): (.25, 2.21),
}
CYLINDRICAL = {'cup', 'paper cup', 'insulated cup', 'thermos', 'bottle',
               'water bottle', 'mug', 'tumbler', 'flask'}
HORIZONTAL_REFERENCES = {'ceiling': 0, 'floor': 1, 'table': 2, 'desk': 3,
                         'tabletop': 4, 'desktop': 5}


def cloud_estimate(points, label, vertical=None, *, truncated=False):
    """Keep a visible box proxy distinct from conditional external closure."""
    envelope = visible_envelope(points)
    result = {
        'status': envelope['status'], 'method': 'trimmed_visible_point_box',
        'estimate_m3': (envelope['envelope_l']/1000
                        if envelope['status'] == 'partial_visible_envelope' else None),
        'dimensions_m': envelope.get('dimensions_m'),
        'visible_envelope': envelope, 'native_samples': len(points),
        'physical_accuracy': 'unverified',
        'semantics': 'visible-point box proxy; unseen surfaces and cavities are not reconstructed',
        'assumptions': ['depth belongs to the detected instance'],
    }
    if label.casefold() in CYLINDRICAL:
        reason = 'truncated_object' if truncated else 'no_vertical_reference' if vertical is None else None
        cylinder = ({'status': reason} if reason else
                    upright_cylinder(points, vertical['normal']))
        result['cylinder'] = cylinder
        if cylinder['status'] == 'conditional_estimate':
            result.update(status='conditional_estimate', method='upright_cylinder',
                          estimate_m3=cylinder['external_envelope_l']/1000,
                          dimensions_m=[cylinder['diameter_m'], cylinder['diameter_m'], cylinder['height_m']],
                          assumptions=cylinder['assumptions'] + [vertical['assumption']],
                          semantics=cylinder['semantics'])
    return result


def estimate_objects(frame, detections, masks):
    """Return JSON-safe, frame-qualified estimates in exactly detection order.

    Masks stay transient. Native pixels remain independent measurements; the
    depth-to-color transform only selects pixels, never transforms dimensions.
    """
    if len(masks) != len(detections):
        raise ValueError('Instance masks must correspond to every detection')
    started = time.monotonic()
    rows = [{
        'schema_version': 1, 'status': 'no_instance_mask', 'method': None,
        'estimate_m3': None, 'dimensions_m': None, 'native_samples': 0,
        'physical_accuracy': 'unverified', 'warnings': [],
        'source_frame_id': frame.id,
        'source_session_id': frame.metadata.get('session_id'),
        'calibration_sha256': frame.metadata.get('calibration_sha256'),
        'source_box': list(d['box']), 'coordinate_frame': 'native_depth_camera',
        'units': 'm3',
    } for d in detections]
    summary = {'method_version': 1, 'basis': 'native_depth_instance_mask',
               'vertical_reference': None}

    def finish(status=None):
        if status:
            for row in rows:
                row.update(status=status, estimate_m3=None, method=None)
        summary['elapsed_ms'] = round((time.monotonic()-started)*1000, 2)
        return rows, summary

    if not rows:
        return finish()
    if frame.depth_native is None:
        return finish('no_native_depth')
    depth = frame.depth_native
    if depth.ndim != 2 or depth.dtype != np.uint16:
        return finish('invalid_depth')
    profile = frame.metadata.get('profiles', {}).get('depth', {})
    shape = (depth.shape[1], depth.shape[0])
    if (profile.get('width'), profile.get('height')) != shape:
        return finish('invalid_depth_profile')
    operating_range = OPERATING_RANGES.get(shape)
    if operating_range is None:
        return finish('unsupported_depth_profile')
    summary['operating_range_m'] = list(operating_range)
    eligible = []
    for index, mask in enumerate(masks):
        if mask is None:
            continue
        if (mask.dtype != np.bool_ or mask.shape != frame.color_rgb.shape[:2]):
            rows[index]['status'] = 'invalid_instance_mask'
            continue
        if not mask.any():
            rows[index]['status'] = 'empty_instance_mask'
            continue
        eligible.append(index)
    if not eligible:
        return finish()
    selected = [masks[index] for index in eligible]
    cores = [cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
             for mask in selected]
    try:
        scale = float(frame.metadata['depth_scale_mm'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid depth scale')
        clean = depth.copy()
        clean[clean == np.iinfo(np.uint16).max] = 0
        maximum = float(clean.max())*scale/1000
        if maximum <= 0:
            for index in eligible:
                rows[index]['status'] = 'insufficient_depth'
            return finish()
        clouds, diagnostic = native_mask_points(
            clean, frame.metadata, selected+cores, min_range=0, max_range=maximum+1e-6)
    except (KeyError, TypeError, ValueError, OverflowError, cv2.error) as exc:
        summary['geometry_error'] = f'{type(exc).__name__}: {str(exc)[:180]}'
        for index in eligible:
            rows[index]['status'] = 'invalid_calibration_or_depth'
        return finish()
    summary['projection'] = diagnostic
    raw_clouds = clouds[:len(eligible)]
    low, high = operating_range
    qualified = [cloud[(cloud[:, 2] >= low) & (cloud[:, 2] <= high)] for cloud in clouds]
    clouds, core_clouds = qualified[:len(eligible)], qualified[len(eligible):]
    # A planar reference supplies an assumed vertical axis, not measured gravity.
    references = sorted(range(len(eligible)), key=lambda j:
                        HORIZONTAL_REFERENCES.get(detections[eligible[j]]['label'].casefold(), 99))
    vertical = None
    for j in references:
        label = detections[eligible[j]]['label'].casefold()
        if label not in HORIZONTAL_REFERENCES or len(clouds[j]) < 100:
            continue
        try:
            plane = fit_plane(clouds[j], threshold_m=.01)
        except UnsupportedPlane:
            continue
        if plane['inlier_fraction'] >= .8 and plane['inlier_residual_p95_m'] <= .02:
            vertical = {**plane, 'box_index': eligible[j], 'label': label,
                        'source_frame_id': frame.id, 'verified': False,
                        'assumption': f'detected {label} is horizontal; its plane normal supplies vertical'}
            break
    summary['vertical_reference'] = vertical
    for j, index in enumerate(eligible):
        mask, cloud = masks[index], clouds[j]
        truncated = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        estimate = cloud_estimate(cloud, detections[index]['label'], vertical, truncated=truncated)
        rows[index].update(estimate, mask_pixels=int(mask.sum()),
                           raw_native_samples=len(raw_clouds[j]),
                           outside_operating_range_samples=len(raw_clouds[j])-len(cloud),
                           operating_range_m=list(operating_range), vertical_reference=vertical)
        if not len(cloud) and len(raw_clouds[j]):
            rows[index]['status'] = 'outside_operating_range'
        if 'calibration_warning' in diagnostic:
            rows[index]['warnings'].append(diagnostic['calibration_warning'])
        if truncated:
            rows[index]['warnings'].append('Object mask reaches image boundary; extent is incomplete')
        if rows[index]['outside_operating_range_samples']:
            rows[index]['warnings'].append('Returns outside the active depth mode range were excluded')
        registered = frame.depth_registered
        if registered is not None and registered.shape == mask.shape:
            valid = mask & (registered > 0) & (registered < np.iinfo(np.uint16).max)
            rows[index]['registered_valid_fraction'] = float(valid.sum()/mask.sum())
        # Sensitivity is a second mask scenario, never an accuracy interval.
        core = cloud_estimate(core_clouds[j], detections[index]['label'], vertical, truncated=truncated)
        rows[index]['mask_sensitivity'] = {
            'erosion_pixels': 3, 'status': core['status'], 'method': core['method'],
            'estimate_m3': core['estimate_m3'], 'native_samples': core['native_samples'],
            'semantics': 'eroded-mask comparison; not confidence bounds',
        }
    return finish()


def volume_label(volume):
    """Compact live overlay text; unavailable values never appear as zero."""
    if not volume:
        return ''
    estimate = volume.get('estimate_m3')
    if estimate is None:
        return 'volume unavailable'
    kind = 'cylinder' if volume.get('method') == 'upright_cylinder' else 'visible box'
    return f'~{estimate*1000:.3g} L ({kind})'
