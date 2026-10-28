#!/usr/bin/env python3
"""Read-only export of archived RGB crops with perceptual groups and held-out sessions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from common.geometry import bounds
from rgb.material_training import group_split


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    temporary = path.with_suffix('.temporary')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def phash(rgb):
    gray = cv2.cvtColor(cv2.resize(rgb, (32, 32)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    values = cv2.dct(gray)[:8, :8].reshape(-1)
    median = np.median(values[1:])
    result = 0
    for flag in values > median:
        result = (result << 1) | int(flag)
    return result


def export_archive(archive, output, maximum=192, minimum_size=32):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output/'dataset.json').exists():
        raise ValueError('Use a new dataset directory; archived evidence is immutable')
    (output/'crops').mkdir()
    (output/'scenes').mkdir()
    connection = sqlite3.connect('file:'+str((archive/'archive.sqlite3').resolve())+'?mode=ro', uri=True)
    connection.execute('BEGIN')
    events = [json.loads(row[0]) for row in connection.execute('select value from observations')]
    connection.close()
    rows, scenes, sessions, identities = [], {}, {}, {}
    for event in events:
        session=event['session_id']
        sessions[session]=min(sessions.get(session,event['observed_at']),event['observed_at'])
    _,session_splits=group_split([],sessions)
    priorities={'train':0,'validation':1,'test':2}
    for event in sorted(events, key=lambda row:row['observed_at']):
        session = event['session_id']
        asset = event['sensor_assets']['color_preview']
        if Path(asset['file']).name != asset['file']:
            raise ValueError('Unsafe archive asset path')
        source = archive/'assets'/asset['file']
        if digest(source) != asset['sha256']:
            raise ValueError('Archive image hash mismatch')
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB'))
        scenes[event['frame_id']] = source
        height, width = rgb.shape[:2]
        for index, detection in enumerate(event['detections']):
            x1, y1, x2, y2 = bounds(detection['box'], width, height)
            if min(x2-x1, y2-y1) < minimum_size:
                continue
            crop = rgb[y1:y2, x1:x2]
            pixels_sha = hashlib.sha256(np.asarray(crop.shape,dtype='<i4').tobytes()+crop.tobytes()).hexdigest()
            occurrence={'frame_id':event['frame_id'],'session_id':session,'event_id':event['event_id'],
                        'observed_at':event['observed_at'],'bounds_xyxy':[x1,y1,x2,y2],
                        'scene_sha256':asset['sha256'],'track_id':detection.get('track_id'),
                        'original_detector_label':detection['label']}
            # Exact pixel duplicates have one immutable training identity.
            if pixels_sha in identities:
                previous=identities[pixels_sha]
                previous['source_observations'].append(event['event_id'])
                previous['source_occurrences'].append(occurrence)
                # Keep the held-out occurrence when identical pixels recur in
                # a later session; deduplication must not hide that session.
                if priorities[session_splits[session]]>=priorities[session_splits[previous['session_id']]]:
                    previous.update(occurrence)
                continue
            sample_id = hashlib.sha256((event['event_id']+':'+str(index)).encode()).hexdigest()[:20]
            image = Image.fromarray(crop)
            image.thumbnail((768, 768))
            path = output/'crops'/f'{sample_id}.png'
            image.save(path)
            row = {'id':sample_id, 'file':str(path.relative_to(output)), 'sha256':digest(path),
                   **occurrence,'source_observations':[event['event_id']],
                   'source_occurrences':[occurrence],'phash':f'{phash(crop):016x}',
                   'aspect_ratio':(x2-x1)/(y2-y1),
                   'mean_rgb':(crop.mean(axis=(0,1))/255).tolist(),
                   }
            rows.append(row)
            identities[pixels_sha] = row
    parents = list(range(len(rows)))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            same_track = (row['track_id'] is not None and row['session_id'] == other['session_id']
                          and row['track_id'] == other['track_id'])
            close = ((int(row['phash'],16) ^ int(other['phash'],16)).bit_count() <= 6
                     and .8 <= row['aspect_ratio']/other['aspect_ratio'] <= 1.25
                     and np.linalg.norm(np.array(row['mean_rgb'])-other['mean_rgb']) <= .15)
            if same_track or close:
                parents[root(i)] = root(j)
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[root(i)].append(row)
    for group in grouped.values():
        identity = hashlib.sha256('\n'.join(sorted(row['id'] for row in group)).encode()).hexdigest()[:20]
        for row in group:
            row['group_id'] = identity
    split_by_group, session_splits = group_split(rows, sessions)
    representatives = []
    for group in grouped.values():
        # Up to two distant views per group; every member remains in one split.
        split = split_by_group[group[0]['group_id']]
        group = [row for row in group if session_splits[row['session_id']] == split]
        group.sort(key=lambda row:row['observed_at'])
        for row in [group[0]]+([group[-1]] if len(group)>1 and group[-1]['phash']!=group[0]['phash'] else []):
            row['split'] = split_by_group[row['group_id']]
            representatives.append(row)
    random.Random(20261028).shuffle(representatives)
    selected = []
    by_split = {split:[r for r in representatives if r['split']==split]
                for split in ['train','validation','test']}
    # Reserve all available held-out representatives before capping train examples.
    for split in ['test','validation','train']:
        cap = maximum//3 if split != 'train' else maximum-len(selected)
        selected.extend(by_split[split][:cap])
    used = {row['id'] for row in selected}
    for row in sorted(selected, key=lambda row:row['id']):
        scene = output/'scenes'/f"{row['frame_id']}.jpg"
        if not scene.exists():
            scene.write_bytes(scenes[row['frame_id']].read_bytes())
        row['scene_file'] = str(scene.relative_to(output))
    # Remove unsampled private pixels from the transferable dataset.
    for row in rows:
        if row['id'] not in used:
            (output/row['file']).unlink()
    dataset = {'schema':1, 'seed':20261028, 'source':'read-only archived camera observations',
               'observations':len(events), 'eligible_unique_crops':len(rows),
               'perceptual_groups':len(grouped), 'selected':len(selected),
               'split_counts':dict(Counter(row['split'] for row in selected)),
               'session_splits':session_splits, 'entries':sorted(selected,key=lambda row:row['id']),
               'grouping':{'phash_max_hamming':6,'aspect_ratio_range':[.8,1.25],'mean_rgb_max_distance':.15},
               'scope':'Approximate visual/track groups, not verified physical-object identities',
               'source_observation_ids_sha256':hashlib.sha256('\n'.join(sorted(e['event_id'] for e in events)).encode()).hexdigest()}
    write(output/'dataset.json', dataset)
    return {key:value for key,value in dataset.items() if key != 'entries'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-crops', type=int, default=192)
    args = parser.parse_args()
    print(json.dumps(export_archive(args.archive, args.output, args.max_crops), indent=2))
