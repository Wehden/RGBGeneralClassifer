#!/usr/bin/env python3
"""Private paired model, segmentation and native-depth qualification receipts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

# Keep unrelated BLAS thread pools from distorting CPU comparisons.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from board.yoloe import Detector, prepare
from common.geometry import bounds
from rgb.instance_masks import instances
from rgb.materials import MaterialClassifier
from rgb.metrology import native_mask_points


def digest(path):
    with Path(path).open('rb') as f:
        result = hashlib.file_digest(f, 'sha256')
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.temporary')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare_minc(archive, output, per_category=20):
    """Deterministic bounded sample; no archive paths are extracted."""
    categories = {label: label for label in
                  ['ceramic', 'fabric', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'wood']}
    categories['water'] = 'liquid'
    heaps = defaultdict(list)
    counts = Counter()
    with tarfile.open(archive, 'r|gz') as members:
        for member in members:
            path = Path(member.name)
            if not member.isfile() or len(path.parts) != 4 or path.parts[1] != 'images':
                continue
            category = path.parts[2]
            if category not in categories or path.suffix.lower() != '.jpg':
                continue
            counts[category] += 1
            key = int(hashlib.sha256(member.name.encode()).hexdigest(), 16)
            selected = heaps[category]
            if len(selected) >= per_category and key >= -selected[0][0]:
                continue
            if member.size > 8*1024*1024:
                raise ValueError('Unexpectedly large MINC image')
            content = members.extractfile(member).read()
            item = (-key, member.name, content)
            if len(selected) < per_category:
                heapq.heappush(selected, item)
            else:
                heapq.heapreplace(selected, item)
    if set(heaps) != set(categories):
        raise ValueError('Missing requested MINC categories: '+str(set(categories)-set(heaps)))
    entries = []
    directory = output/'minc-images'
    directory.mkdir(parents=True, exist_ok=True)
    for category, selected in sorted(heaps.items()):
        if len(selected) != per_category:
            raise ValueError('Insufficient MINC samples')
        for index, (_, source, content) in enumerate(sorted(selected, key=lambda item: item[1])):
            image = Image.open(io.BytesIO(content)).convert('RGB')
            filename = directory/f'{category}-{index:03}.png'
            image.save(filename)
            entries.append({'file':str(filename.resolve()), 'sha256':digest(filename),
                            'ground_truth':categories[category], 'source_category':category,
                            'source_image':source, 'kind':'MINC-2500'})
    dataset = {'source':'https://opensurfaces.cs.cornell.edu/publications/minc/',
               'archive_sha256':digest(archive), 'samples_per_category':per_category,
               'selection':'20 lowest SHA256(source image name) per selected category',
               'scope':'zero-shot pilot; no official train/test split filtering or camera-domain accuracy claim',
               'source_category_counts':dict(counts), 'images':entries}
    write_json(output/'minc-dataset.json', dataset)
    return {'images':len(entries), 'categories':len(categories), 'dataset':str(output/'minc-dataset.json')}


def infer(model, bgr):
    tensor, transform = prepare(bgr, model.size)
    start = time.perf_counter()
    outputs = model.session.run(None, {model.input_name:tensor})
    inference_ms = (time.perf_counter()-start)*1000
    readings = instances(outputs[0], outputs[1], model.labels, bgr.shape, transform, model.threshold)
    return readings, outputs, inference_ms


def mask_reference(readings, outputs, shape):
    """Check NumPy decoder against the pinned upstream mask implementation."""
    if not readings:
        return {'instances':0, 'minimum_mask_iou':None}
    import torch
    from ultralytics.utils.ops import process_mask_native
    torch.set_num_threads(4)
    h, w = shape[:2]
    rows = [item['row_index'] for item in readings]
    boxes = [[item['box'][1]*w, item['box'][0]*h,
              item['box'][3]*w, item['box'][2]*h] for item in readings]
    source = process_mask_native(torch.from_numpy(outputs[1][0]),
                                 torch.from_numpy(outputs[0][0, rows, 6:]),
                                 torch.tensor(boxes, dtype=torch.float32), (h, w)).numpy().astype(bool)
    ious = []
    for reading, reference in zip(readings, source):
        union = np.count_nonzero(reading['mask'] | reference)
        iou = np.count_nonzero(reading['mask'] & reference)/union if union else 1.0
        ious.append(float(iou))
    if min(ious) < .999:
        raise ValueError(f'Mask reference parity failed: minimum IoU {min(ious)}')
    return {'instances':len(readings), 'minimum_mask_iou':min(ious),
            'reference':'ultralytics==8.4.188 process_mask_native'}


def detector_worker(directory, frames, output, samples, threads):
    model = Detector(directory/'model.onnx', directory/'manifest.json', threads=threads)
    bundles = []
    for path in frames:
        with np.load(path, allow_pickle=False) as bundle:
            bundles.append((bundle['color_rgb'].copy(), bundle['depth_native'].copy(),
                            bundle['depth_registered'].copy(),
                            json.loads(bundle['metadata_json_utf8'].tobytes())))
    bgrs = [cv2.cvtColor(bundle[0], cv2.COLOR_RGB2BGR) for bundle in bundles]
    for index in range(5):
        infer(model, bgrs[index % len(bgrs)])
    total = []; inference = []; saved = []
    for index in range(max(samples, len(bundles))):
        start = time.perf_counter()
        readings, outputs, ms = infer(model, bgrs[index % len(bgrs)])
        if index < samples:
            total.append((time.perf_counter()-start)*1000);inference.append(ms)
        if index < len(bundles):
            saved.append((readings, outputs))
    scene_readings = []; parity = []; geometry_timings = []; camera_crops = []
    for index, ((rgb, depth, registered, metadata), (readings, outputs)) in enumerate(zip(bundles, saved, strict=True)):
        parity.append(mask_reference(readings, outputs, rgb.shape))
        start = time.perf_counter()
        clouds, diagnostics = native_mask_points(depth, metadata, [item['mask'] for item in readings])
        geometry_timings.append((time.perf_counter()-start)*1000)
        rows = []
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for item_index, (reading, points) in enumerate(zip(readings, clouds)):
            mask = reading['mask']
            valid_registered = mask & (registered > 0)
            row = {key:value for key,value in reading.items() if key != 'mask'}
            row.update(mask_pixels=int(mask.sum()), native_depth_samples=len(points),
                       registered_valid_fraction=float(valid_registered.sum()/mask.sum()) if mask.any() else 0)
            if len(points) >= 30:
                low, high = np.percentile(points, [1, 99], axis=0)
                row['observed_cloud_bounds_m'] = (high-low).tolist()
                row['bounds_semantics'] = 'native-camera axes, central 98% of visible points; not full object dimensions or volume'
            rows.append(row)
            overlay[mask] = (overlay[mask].astype(np.float32)*.65+np.array([60, 220, 60])*.35).astype(np.uint8)
            if index == 0 and directory.name == 'yoloe-26s-pf-640' and item_index < 8:
                x1,y1,x2,y2 = bounds(reading['box'], rgb.shape[1], rgb.shape[0])
                if min(x2-x1,y2-y1) < 16:
                    continue
                crop = rgb[y1:y2,x1:x2].copy()
                foreground = crop.copy()
                foreground[~mask[y1:y2,x1:x2]] = 127
                for kind, pixels in [('box',crop),('masked',foreground)]:
                    path = output/'camera-crops'/f'object-{item_index:02}-{kind}.png'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(pixels).save(path)
                    camera_crops.append({'file':str(path.resolve()),'sha256':digest(path),
                                         'ground_truth':None,'kind':kind,'frame_id':metadata['frame_id'],
                                         'box':reading['box'],'detector_label':reading['label']})
        path = output/'overlays'/f'{directory.name}-scene-{index:02}.jpg'
        path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(path), overlay)
        scene_readings.append({'source_file':str(frames[index]),'frame_id':metadata['frame_id'],
                               'detections':rows,'native_depth_diagnostics':diagnostics})
    if camera_crops:
        write_json(output/'camera-crops.json', {'images':camera_crops,'ground_truth':'unavailable'})
    # Reference libraries increase RSS; report them separately from the measured inference process.
    return {'model':directory.name,'config':model.config,'samples':samples,'frames':len(frames),
            'total_p50_ms':float(np.median(total)),'total_p95_ms':float(np.percentile(total,95)),
            'inference_p50_ms':float(np.median(inference)),
            'geometry_p50_ms':float(np.median(geometry_timings)),
            'peak_rss_including_reference_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mask_reference_parity':parity,'scene_readings':scene_readings,
            'object_accuracy':'unmeasured: camera scenes have no verified object labels'}


def material_worker(directory, dataset_paths, samples, threads):
    classifier = MaterialClassifier(directory, threads=threads)
    entries = []
    for path in dataset_paths:
        document = json.loads(path.read_text())
        for entry in document['images']:
            if digest(entry['file']) != entry['sha256']:
                raise ValueError('Dataset image checksum mismatch')
            entries.append(entry)
    arrays = [np.asarray(Image.open(entry['file']).convert('RGB')) for entry in entries]
    for index in range(5):
        classifier.classify(arrays[index % len(arrays)])
    timings = []; readings = []; by_kind = defaultdict(list)
    for index, (entry, rgb) in enumerate(zip(entries, arrays)):
        start = time.perf_counter()
        result = classifier.classify(rgb)
        elapsed = (time.perf_counter()-start)*1000
        timings.append(elapsed);by_kind[entry['kind']].append(elapsed)
        readings.append({**entry,'status':result['status'],'label':result['label'],
                         'best_candidate':result['best_candidate'],'similarity':result['similarity'],
                         'margin':result['margin'],'candidates':result['candidates'],'total_ms':elapsed})
    # Repeat an identical crop for a stable warm-latency comparison.
    repeated = []
    for _ in range(samples):
        start = time.perf_counter();classifier.classify(arrays[0])
        repeated.append((time.perf_counter()-start)*1000)
    labeled = [item for item in readings if item['ground_truth'] is not None]
    accepted = [item for item in labeled if item['status'] == 'identified']
    correct = sum(item['best_candidate'] == item['ground_truth'] for item in labeled)
    per_class = {}
    for label in sorted({item['ground_truth'] for item in labeled}):
        rows = [item for item in labeled if item['ground_truth'] == label]
        per_class[label] = {'images':len(rows),
                            'top1_accuracy':sum(item['best_candidate']==label for item in rows)/len(rows),
                            'accepted':sum(item['status']=='identified' for item in rows)}
    return {'model':directory.name,'config':classifier.config,'images':len(entries),
            'total_p50_ms':float(np.median(timings)),'total_p95_ms':float(np.percentile(timings,95)),
            'fixed_crop_p50_ms':float(np.median(repeated)),
            'fixed_crop_p95_ms':float(np.percentile(repeated,95)),
            'by_kind':{kind:{'images':len(values),'p50_ms':float(np.median(values)),
                             'p95_ms':float(np.percentile(values,95))} for kind,values in by_kind.items()},
            'labeled_images':len(labeled),'top1_accuracy':correct/len(labeled) if labeled else None,
            'acceptance_coverage':len(accepted)/len(labeled) if labeled else None,
            'accepted_label_precision':sum(item['label']==item['ground_truth'] for item in accepted)/len(accepted) if accepted else None,
            'per_material':per_class,'readings':readings,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'policy':'existing 0.20 similarity / 0.02 margin; uncalibrated across encoders'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--models-dir',type=Path,default=Path.home()/'.cache/bolt-rgb/models')
    parser.add_argument('--prepare-minc',type=Path)
    parser.add_argument('--stage',choices=['detectors','materials'])
    parser.add_argument('--frames',nargs='+',type=Path)
    parser.add_argument('--datasets',nargs='+',type=Path)
    parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--threads',type=int)
    parser.add_argument('--worker',nargs=2,metavar=('KIND','DIRECTORY'))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.prepare_minc:
        print(json.dumps(prepare_minc(args.prepare_minc,args.output_dir)),flush=True)
        return
    if args.worker:
        kind, directory = args.worker;directory = Path(directory)
        if kind == 'detectors':
            result = detector_worker(directory,args.frames,args.output_dir,args.samples,args.threads or 4)
        else:
            result = material_worker(directory,args.datasets,args.samples,args.threads or 2)
        print(json.dumps(result,allow_nan=False))
        return
    if not args.stage or (args.stage=='detectors' and not args.frames) or (args.stage=='materials' and not args.datasets):
        parser.error('select a stage and provide its frame/dataset inputs')
    names = (['yoloe-26s-pf-640','yoloe-26m-pf-640','yoloe-26l-pf-640'] if args.stage=='detectors'
             else ['siglip2-base-224','siglip2-base-384','siglip2-large-256'])
    results = []
    for name in names:
        command = [sys.executable,__file__,'--output-dir',str(args.output_dir),
                   '--worker',args.stage,str(args.models_dir/name),'--samples',str(args.samples)]
        if args.threads:
            command += ['--threads',str(args.threads)]
        if args.frames:
            command += ['--frames',*map(str,args.frames)]
        if args.datasets:
            command += ['--datasets',*map(str,args.datasets)]
        process = subprocess.run(command,capture_output=True,text=True)
        if process.returncode:
            result = {'model':name,'state':'failed','error':process.stderr[-4000:]}
        else:
            result = json.loads(process.stdout)
            result['state'] = 'passed'
        results.append(result)
        write_json(args.output_dir/f'{name}.json',result)
        print(json.dumps({key:value for key,value in result.items()
                          if key in ['model','state','error','total_p50_ms','total_p95_ms',
                                     'top1_accuracy','acceptance_coverage','accepted_label_precision']}),flush=True)
    write_json(args.output_dir/f'{args.stage}-summary.json',{'stage':args.stage,'results':results})
    if any(result['state']!='passed' for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
