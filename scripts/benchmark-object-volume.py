#!/usr/bin/env python3
"""Replay retained frame bundles through the live per-object volume estimator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board.yoloe import Detector
from rgb.camera import Frame
from rgb.object_volume import estimate_objects, volume_label


def analyse(path, model, output):
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(bundle['metadata_json_utf8'].tobytes())
        frame = Frame(metadata, bundle['color_rgb'].copy(),
                      bundle['depth_native'].copy() if 'depth_native' in bundle else None,
                      bundle['ir_native'].copy() if 'ir_native' in bundle else None,
                      bundle['depth_registered'].copy() if 'depth_registered' in bundle else None, {})
    start = time.monotonic()
    bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    detections, inference_ms = model.detect(bgr, with_masks=frame.depth_native is not None)
    mask_ms = round((time.monotonic()-start)*1000, 2)
    if frame.depth_native is not None:
        volumes, summary = estimate_objects(frame, detections, model.instance_masks)
        for d, volume in zip(detections, volumes):
            d['volume'] = volume
    else:
        summary = None
    receipt = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'frame_id': frame.id, 'calibration_sha256': metadata.get('calibration_sha256'),
               'detector_sha256': model.sha256, 'detector_config': model.config,
               'inference_ms': inference_ms, 'detection_with_masks_ms': mask_ms,
               'mask_error': model.mask_error, 'metrology': summary, 'detections': detections}
    (output/(path.stem+'.json')).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    from common.geometry import bounds
    for d in detections:
        x1,y1,x2,y2 = bounds(d['box'], bgr.shape[1], bgr.shape[0])
        label = d['label']
        text = volume_label(d.get('volume'))
        if text:
            label += ' / '+text
        cv2.rectangle(bgr, (x1,y1), (x2,y2), (50,220,90), 2)
        cv2.putText(bgr, label, (x1,max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (50,220,90), 1)
    if not cv2.imwrite(str(output/(path.stem+'.jpg')), bgr):
        raise ValueError('Could not write object overlay')
    print(json.dumps({'frame_id': frame.id, 'boxes': len(detections), 'inference_ms': inference_ms,
                      'detection_with_masks_ms': mask_ms,
                      'metrology_ms': summary['elapsed_ms'] if summary else None}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, nargs='+', required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=['onnxruntime', 'openvino'], default='onnxruntime')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new empty output directory to preserve earlier measurements')
    if len({p.stem for p in args.frames}) != len(args.frames):
        raise ValueError('Frame filenames must have distinct stems')
    if not 1 <= args.threads <= 64:
        raise ValueError('Invalid thread count')
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    model = Detector(args.model_dir/'model.onnx', args.model_dir/'manifest.json',
                     backend=args.backend, threads=args.threads)
    for path in args.frames:
        analyse(path, model, args.output_dir)


if __name__ == '__main__':
    main()
