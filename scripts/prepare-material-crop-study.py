#!/usr/bin/env python3
"""Prepare paired crop strategies from one immutable capture; no cloud calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board.yoloe import Detector, prepare
from rgb.instance_masks import instances
from rgb.material_crops import material_crop


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Use a new immutable study directory')
    capture = np.load(args.capture, allow_pickle=False)
    rgb = capture['color_rgb']
    detector = Detector(args.detector/'model.onnx', args.detector/'manifest.json',
                        threshold=args.threshold, threads=4)
    tensor, transform = prepare(rgb[:, :, ::-1], detector.size)
    outputs = detector.session.run(None, {detector.input_name: tensor})
    found = instances(*outputs, detector.labels, rgb.shape, transform, args.threshold)
    args.output.mkdir(parents=True, mode=0o700)
    (args.output/'crops').mkdir()
    Image.fromarray(rgb).save(args.output/'scene.jpg', quality=92)
    h, w = rgb.shape[:2]
    rows, excluded, groups = [], [], []
    per_label = Counter()
    for row in sorted(found, key=lambda row: -row['confidence']):
        if len(groups) >= args.max_objects:
            break
        # This is a surface crop comparison, not a human identity classifier.
        if row['label'] in {'person', 'glasses', 'ceiling', 'chain mail'}:
            continue
        if per_label[row['label']] >= 2:
            continue
        y1, x1, y2, x2 = row['box']
        bounds = [int(np.floor(x1*w)), int(np.floor(y1*h)),
                  min(w, int(np.ceil(x2*w))), min(h, int(np.ceil(y2*h)))]
        variants = []
        try:
            for strategy in ['stretch', 'letterbox', 'foreground', 'interior']:
                pixels, context = material_crop(rgb, bounds, strategy, row['mask'])
                variants.append((strategy, pixels, context))
        except ValueError as error:
            excluded.append({'row_index': row['row_index'], 'reason': str(error)})
            continue
        group_id = f"capture-{digest(args.capture)[:16]}-object-{row['row_index']}"
        groups.append(group_id)
        per_label[row['label']] += 1
        for strategy, pixels, context in variants:
            sample_id = group_id+'-'+strategy
            relative = 'crops/'+sample_id+'.jpg'
            Image.fromarray(pixels).save(args.output/relative, quality=92)
            rows.append({'id': sample_id, 'group_id': group_id,
                         'split': 'evaluation-candidate', 'strategy': strategy,
                         'file': relative, 'sha256': digest(args.output/relative),
                         'scene_file': 'scene.jpg',
                         'scene_sha256': digest(args.output/'scene.jpg'),
                         'bounds_xyxy': context, 'object_bounds_xyxy': bounds,
                         'source_width_pixels': context[2]-context[0],
                         'source_height_pixels': context[3]-context[1],
                         'mask_foreground_fraction': float(
                             row['mask'][context[1]:context[3],context[0]:context[2]].mean()),
                         'detector_label': row['label'], 'detector_confidence': row['confidence']})
    protocol = {
        'schema_version': 1, 'capture_sha256': digest(args.capture),
        'detector': detector.config, 'threshold': args.threshold,
        'maximum_objects': args.max_objects, 'minimum_interior_side_pixels': 24,
        'selection': 'confidence order; max two of each detector label; all four variants required',
        'student_pixels': '384x384 RGB JPEG; labeler must preserve exact source JPEG bytes',
        'teacher': 'two blinded Nova 2 Lite passes with exact variant crop and annotated scene; unique-groups-per-batch required',
        'analysis': 'paired descriptive teacher agreement and material resolution; no model selection',
        'identity': 'same-frame detector groups; physical novelty and composition unverified',
        'eligibility': 'not a fresh confirmatory object-disjoint holdout',
        'entries': rows, 'excluded': excluded}
    (args.output/'dataset.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'objects': len(groups), 'samples': len(rows),
                      'excluded': len(excluded), 'dataset_sha256': digest(args.output/'dataset.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-objects', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=.65)
    args = parser.parse_args()
    if args.max_objects < 1 or not 0 <= args.threshold <= 1:
        parser.error('Invalid selection bounds')
    run(args)
