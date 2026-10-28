#!/usr/bin/env python3
"""Blinded two-pass image reviews of saved detector boxes, with private receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from common.geometry import bounds
from rgb import agreement
from rgb.vlm_providers import BedrockProvider, PRIMARY_MODEL, ReviewRequest


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.temporary')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def jpeg(image):
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=92, subsampling=0)
    return stream.getvalue()


def prepare(receipts, output, scene_indices):
    """Deduplicate only exactly identical crop pixels; retain every prediction."""
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    evidence = output/'evidence'
    evidence.mkdir(exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest['matching_implementation_sha256'] != agreement.IMPLEMENTATION_SHA256
                or manifest['system'] != agreement.SYSTEM):
            raise ValueError('Review protocol changed; use a new output directory')
        if manifest['scene_indices'] != scene_indices:
            raise ValueError('Scene selection changed; use a new output directory')
        if manifest['detector_receipts'] != [
                {'file':str(p.resolve()), 'sha256':digest(p)} for p in receipts]:
            raise ValueError('Detector receipts changed; use a new output directory')
        for source in manifest['sources']:
            if digest(source['file']) != source['sha256']:
                raise ValueError('Source frame changed')
        for batch in manifest['batches']:
            if batch['prompt'] != agreement.prompt(len(batch['case_ids'])):
                raise ValueError('Review prompt changed')
            for image in batch['images']:
                if digest(image['file']) != image['sha256']:
                    raise ValueError('Review image changed')
        return manifest
    cases, sources, by_bounds = [], {}, {}
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        model = receipt_path.stem
        for scene_index in scene_indices:
            scene = receipt['scene_readings'][scene_index]
            source = Path(scene['source_file'])
            with np.load(source, allow_pickle=False) as frame:
                rgb = frame['color_rgb']
                metadata = json.loads(frame['metadata_json_utf8'].tobytes())
                if metadata['frame_id'] != scene['frame_id']:
                    raise ValueError('Detector/frame identity mismatch')
                height, width = rgb.shape[:2]
                sources.setdefault(scene['frame_id'], {'file':str(source.resolve()),
                    'frame_id':scene['frame_id'], 'sha256':digest(source),
                    'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
                for detection in scene['detections']:
                    rectangle = bounds(detection['box'], width, height)
                    x1, y1, x2, y2 = rectangle
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError('Empty detector box')
                    key = (scene['frame_id'], rectangle)
                    if key not in by_bounds:
                        crop = Image.fromarray(rgb[y1:y2, x1:x2])
                        crop.thumbnail((512, 512))
                        case_id = f'case-{len(cases):03}'
                        filename = evidence/f'{case_id}.jpg'
                        filename.write_bytes(jpeg(crop))
                        case = {'case_id':case_id, 'frame_id':scene['frame_id'],
                                'scene_index':scene_index, 'bounds_xyxy':list(rectangle),
                                'crop_file':str(filename.resolve()), 'crop_sha256':digest(filename),
                                'predictions':[]}
                        cases.append(case)
                        by_bounds[key] = case
                    by_bounds[key]['predictions'].append({
                        'model':model, 'label':detection['label'],
                        'confidence':detection['confidence'], 'box':detection['box'],
                        'row_index':detection['row_index']})
    batches = []
    colors = ['#ff2d55', '#00ffff', '#ffff00', '#30ff40']
    for frame_id, source in sources.items():
        group = [case for case in cases if case['frame_id'] == frame_id]
        random.Random(20261028).shuffle(group)
        with np.load(source['file'], allow_pickle=False) as frame:
            rgb = Image.fromarray(frame['color_rgb'])
        for start in range(0, len(group), 4):
            selected = group[start:start+4]
            batch_id = f'batch-{len(batches):03}'
            scene = rgb.copy()
            draw = ImageDraw.Draw(scene)
            for index, case in enumerate(selected):
                x1, y1, x2, y2 = case['bounds_xyxy']
                draw.rectangle((x1, y1, x2-1, y2-1), outline=colors[index], width=3)
                tx, ty = min(x1, scene.width-24), max(0, y1-24)
                draw.rectangle((tx, ty, tx+22, ty+22), fill=colors[index])
                draw.text((tx+6, ty+3), str(index), fill='black', font_size=16)
            filename = evidence/f'{batch_id}-scene.jpg'
            filename.write_bytes(jpeg(scene))
            images = [{'label':'Scene context with numbered rectangles', 'file':str(filename.resolve()),
                       'sha256':digest(filename)}]
            images.extend({'label':f'Unannotated crop for rectangle {index}',
                           'file':case['crop_file'], 'sha256':case['crop_sha256']}
                          for index, case in enumerate(selected))
            batches.append({'batch_id':batch_id, 'case_ids':[c['case_id'] for c in selected],
                            'prompt':agreement.prompt(len(selected)), 'images':images})
    manifest = {'schema':1, 'created_at':datetime.now(timezone.utc).isoformat(),
                'matching_implementation_sha256':agreement.IMPLEMENTATION_SHA256,
                'matching_policy':agreement.OBJECT_ALIASES, 'system':agreement.SYSTEM,
                'scene_indices':scene_indices, 'sources':list(sources.values()),
                'detector_receipts':[{'file':str(p.resolve()), 'sha256':digest(p)} for p in receipts],
                'cases':cases, 'batches':batches, 'passes':2,
                'design':'Two full blinded sweeps; identical prompts/images, separate calls, temperature zero',
                'scope':'Candidate-box precision proxy; no recall, ground-truth accuracy or localization IoU claim'}
    write_json(manifest_path, manifest)
    return manifest


def collect(manifest, output):
    cases = [dict(case, reviews=[None, None]) for case in manifest['cases']]
    lookup = {case['case_id']:case for case in cases}
    requests, errors, input_tokens, output_tokens = 0, 0, 0, 0
    for pass_index in range(2):
        for batch in manifest['batches']:
            path = output/'requests'/f"{batch['batch_id']}-pass-{pass_index+1}.json"
            if not path.exists():
                continue
            receipt = json.loads(path.read_text())
            requests += 1
            input_tokens += receipt.get('audit', {}).get('prompt_tokens') or 0
            output_tokens += receipt.get('audit', {}).get('output_tokens') or 0
            if receipt['status'] != 'ok':
                errors += 1
                continue
            reviews = agreement.parse_review(receipt['response_text'], len(batch['case_ids']))
            for index, case_id in enumerate(batch['case_ids']):
                lookup[case_id]['reviews'][pass_index] = reviews[index]
    summary = agreement.summarize(cases)
    summary.update(unique_cases=len(cases), scenes=len(manifest['sources']),
                   calls_recorded=requests, calls_planned=len(manifest['batches'])*2,
                   calls_failed_or_incomplete=errors, input_tokens=input_tokens, output_tokens=output_tokens)
    summary['per_scene'] = {
        str(index):agreement.summarize([case for case in cases if case['scene_index'] == index])['models']
        for index in manifest['scene_indices']}
    per_pass = []
    for pass_index in range(2):
        models = {}
        for case in cases:
            review = case['reviews'][pass_index]
            for prediction in case['predictions']:
                bucket = models.setdefault(prediction['model'],
                    {'detections':0, 'reviewed':0, 'evaluable':0, 'label_confirmed':0,
                     'fit_confirmed':0, 'joint_confirmed':0})
                bucket['detections'] += 1
                if review is None:
                    continue
                bucket['reviewed'] += 1
                clear = agreement.identifiable(review)
                label = agreement.matches(prediction['label'], review)
                fit = clear and review['box_fit'] == 'good'
                bucket['evaluable'] += int(clear)
                bucket['label_confirmed'] += int(label)
                bucket['fit_confirmed'] += int(fit)
                bucket['joint_confirmed'] += int(label and fit)
        per_pass.append(models)
    summary['per_pass'] = per_pass
    consensus = {'reviewed_twice':0, 'label_evaluable':0, 'same_primary_object_label':0,
                 'same_fit':0, 'both_clear_same_material':0}
    for case in cases:
        score = agreement.score_case('', case['reviews'])
        consensus['reviewed_twice'] += int(score['reviewed_twice'])
        consensus['label_evaluable'] += int(score['evaluable'])
        consensus['same_primary_object_label'] += int(score['reviewer_label_agreement'] is True)
        consensus['same_fit'] += int(score['reviewer_fit_agreement'] is True)
        consensus['both_clear_same_material'] += int(score['material_consensus'] is not None)
    summary['reviewer_consensus_unique_cases'] = consensus
    write_json(output/'cases.json', cases)
    write_json(output/'summary.json', summary)
    return summary


def run(manifest, output, profile, region, model, limit_calls):
    directory = output/'requests'
    directory.mkdir(exist_ok=True)
    provider = BedrockProvider(profile=profile, region=region, read_timeout=60)
    completed = 0
    for pass_index in range(2):
        for batch in manifest['batches']:
            path = directory/f"{batch['batch_id']}-pass-{pass_index+1}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if (previous['model'], previous['profile'], previous['region']) != (model, profile, region):
                    raise ValueError('Reviewer configuration changed; use a new output directory')
                # A started receipt could represent a billed but interrupted request; never replay it.
                continue
            if completed >= limit_calls:
                return collect(manifest, output)
            images = tuple((image['label'], Path(image['file']).read_bytes()) for image in batch['images'])
            receipt = {'batch_id':batch['batch_id'], 'pass':pass_index+1,
                       'model':model, 'profile':profile, 'region':region,
                       'temperature':0, 'max_tokens':2400, 'status':'started',
                       'started_at':datetime.now(timezone.utc).isoformat(),
                       'prompt_sha256':hashlib.sha256(batch['prompt'].encode()).hexdigest(),
                       'system_sha256':hashlib.sha256(manifest['system'].encode()).hexdigest(),
                       'matching_implementation_sha256':manifest['matching_implementation_sha256'],
                       'images':[{'label':image['label'], 'sha256':image['sha256']} for image in batch['images']]}
            write_json(path, receipt)
            started = time.monotonic()
            try:
                response = provider.review(ReviewRequest(model, manifest['system'], batch['prompt'], images, 2400))
                receipt.update(response_text=response.text, audit=response.audit)
                agreement.parse_review(response.text, len(batch['case_ids']))
                receipt['status'] = 'ok'
            except Exception as exc:
                receipt.update(status='error', error_type=type(exc).__name__)
                if hasattr(exc, 'audit'):
                    receipt['audit'] = exc.audit
                if isinstance(exc, ValueError):
                    receipt['validation_error'] = str(exc)
            receipt['elapsed_s'] = time.monotonic()-started
            write_json(path, receipt)
            completed += 1
            print(f"{batch['batch_id']} pass {pass_index+1}: {receipt['status']}", flush=True)
    return collect(manifest, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-indices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--invoke-bedrock', action='store_true', help='Invoke up to two image reviews per batch')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--region', default='us-west-2')
    parser.add_argument('--model', default=PRIMARY_MODEL)
    parser.add_argument('--limit-calls', type=int, default=64)
    args = parser.parse_args()
    manifest = prepare(args.receipts, args.output, args.scene_indices)
    result = (run(manifest, args.output, args.profile, args.region, args.model, args.limit_calls)
              if args.invoke_bedrock else collect(manifest, args.output))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
