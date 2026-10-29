#!/usr/bin/env python3
"""Read retained RGB/depth bundles and write private range/volume experiment receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board.yoloe import Detector, prepare
from rgb.capture_metrology import observed_range, upright_cylinder, visible_envelope
from rgb.instance_masks import instances
from rgb.metrology import backproject, fit_plane, native_mask_points


def analyse(path, model, output):
    with np.load(path, allow_pickle=False) as bundle:
        rgb = bundle['color_rgb'].copy()
        depth = bundle['depth_native'].copy()
        registered = bundle['depth_registered'].copy()
        metadata = json.loads(bundle['metadata_json_utf8'].tobytes())
    scale = metadata['depth_scale_mm']
    range_report = observed_range(depth, scale)
    profile = metadata['profiles']['depth']
    if (profile['width'], profile['height']) == (640, 576):
        metres = depth.astype(np.float64)*scale/1000
        returned = (depth > 0) & (depth < np.iinfo(np.uint16).max)
        outside = returned & ((metres < .5) | (metres > 3.86))
        range_report['mode_operating_range'] = {
            'mode': 'NFOV unbinned', 'min_m': .5, 'max_m': 3.86,
            'inferred_from_native_profile': True,
            'outside_samples': int(outside.sum()),
            'outside_fraction_of_returns': float(outside.sum()/max(1, returned.sum())),
            'source': 'https://www.orbbec.com/documentation/femto-bolt-hardware-specifications/'}
    depth[depth == np.iinfo(np.uint16).max] = 0
    maximum = float(depth.max())*scale/1000 + 1e-6
    if maximum <= 1e-6:
        raise ValueError('Frame contains no native depth')
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tensor, transform = prepare(bgr, model.size)
    tensors = model.session.run(None, {model.input_name: tensor})
    detections = instances(tensors[0], tensors[1], model.labels, rgb.shape, transform, model.threshold)
    masks = [d['mask'] for d in detections]
    clouds, diagnostic = native_mask_points(depth, metadata, masks, min_range=0, max_range=maximum)
    all_points, _ = backproject(depth, metadata['calibration']['planes']['depth'], scale,
                               min_range=0, max_range=maximum)
    radial = np.linalg.norm(all_points, axis=1)
    gravity = None
    for detection, cloud in zip(detections, clouds):
        operating = range_report.get('mode_operating_range')
        if operating:
            cloud = cloud[(cloud[:, 2] >= operating['min_m']) & (cloud[:, 2] <= operating['max_m'])]
        if detection['label'] == 'ceiling' and len(cloud) >= 100:
            candidate = fit_plane(cloud, threshold_m=.01)
            if candidate['inlier_fraction'] >= .8 and candidate['inlier_residual_p95_m'] <= .02:
                gravity = {**candidate, 'assumption': 'detected ceiling is horizontal; its normal supplies vertical',
                           'samples': len(cloud), 'operating_range_filter_applied': operating is not None,
                           'verified': False}
                break
    core_masks = [cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) for mask in masks]
    cores, _ = native_mask_points(depth, metadata, core_masks, min_range=0, max_range=maximum)
    overlay = bgr.copy()
    rows = []
    cylindrical = {'cup', 'paper cup', 'insulated cup', 'thermos', 'bottle', 'water bottle', 'mug', 'tumbler', 'flask'}
    for index, (detection, cloud, core) in enumerate(zip(detections, clouds, cores)):
        mask = detection['mask']
        values = registered[mask & (registered > 0)]*metadata['registered_depth_scale_mm']/1000
        row = {key: value for key, value in detection.items() if key != 'mask'}
        row.update(index=index, mask_pixels=int(mask.sum()), native_samples=len(cloud),
                   registered_valid_fraction=float(len(values)/max(1, mask.sum())),
                   visible_envelope=visible_envelope(cloud))
        if len(cloud):
            row['axial_range_m'] = dict(zip(['p02', 'median', 'p98'], np.percentile(cloud[:, 2], [2, 50, 98]).tolist()))
            operating = range_report.get('mode_operating_range')
            if operating:
                row['outside_mode_operating_range'] = bool(
                    row['axial_range_m']['p02'] < operating['min_m']
                    or row['axial_range_m']['p98'] > operating['max_m'])
        if len(values):
            row['registered_axial_median_m'] = float(np.median(values))
        if gravity and detection['label'] in cylindrical:
            if row.get('outside_mode_operating_range'):
                row['cylinder'] = row['eroded_mask_cylinder'] = {'status': 'outside_mode_operating_range'}
            else:
                row['cylinder'] = upright_cylinder(cloud, gravity['normal'])
                row['eroded_mask_cylinder'] = upright_cylinder(core, gravity['normal'])
            scenarios = [r['external_envelope_l'] for r in [row['cylinder'], row['eroded_mask_cylinder']]
                         if r['status'] == 'conditional_estimate']
            if len(scenarios) == 2:
                row['mask_sensitivity_l'] = [min(scenarios), max(scenarios)]
                row['mask_sensitivity_semantics'] = 'two mask scenarios; not an uncertainty/confidence interval'
        row['volume_status'] = row.get('cylinder', {}).get('status', 'no_complete_volume_estimate')
        overlay[mask] = (overlay[mask].astype(float)*.8 + np.array([0, 255, 0])*.2).astype(np.uint8)
        y1, x1, y2, x2 = detection['box'];h, w = rgb.shape[:2]
        cv2.rectangle(overlay, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 1)
        cv2.putText(overlay, f'{index}: {detection["label"]}', (int(x1*w), max(15, int(y1*h))),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
        rows.append(row)
    receipt = {'frame_id': metadata['frame_id'], 'captured_at': metadata['received_at'],
               'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'calibration_sha256': metadata['calibration_sha256'],
               'detector': {'model': model.model, 'sha256': model.sha256, 'threshold': model.threshold},
               'range': range_report,
               'radial_range_m': dict(zip(['min', 'p01', 'median', 'p99', 'max'],
                                        np.percentile(radial, [0, 1, 50, 99, 100]).tolist())),
               'geometry_diagnostics': diagnostic, 'vertical_reference': gravity, 'objects': rows,
               'total_object_volume': None,
               'total_volume_reason': 'incomplete/overlapping views and unsupported object closure; no defensible sum',
               'physical_accuracy': 'unverified'}
    (output/(path.stem+'-measurements.json')).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    cv2.imwrite(str(output/(path.stem+'-objects.jpg')), overlay)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, nargs='+', required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    model = Detector(args.model_dir/'model.onnx', args.model_dir/'manifest.json', threads=4)
    receipts = [analyse(path, model, args.output_dir) for path in args.frames]
    summary = {'frames': len(receipts), 'observed_axial_min_m': min(r['range']['min_m'] for r in receipts),
               'observed_axial_max_m': max(r['range']['max_m'] for r in receipts),
               'measurements': [r['frame_id'] for r in receipts]}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
