#!/usr/bin/env python3
"""Describe paired crop agreement with weak teachers using the deployed CPU runtime."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgb.material_training import LABELS, REJECTIONS
from rgb.materials import MaterialClassifier

STRATEGIES = ('stretch', 'letterbox', 'foreground', 'interior')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence_path(directory, filename, expected):
    path = (directory/filename).resolve()
    if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
        raise ValueError('Crop-study evidence path or checksum mismatch')
    return path


def load_study(directory):
    dataset = json.loads((directory/'dataset.json').read_text())
    labels = json.loads((directory/'teacher-labels.json').read_text())
    dataset_hash = digest(directory/'dataset.json')
    if labels.get('dataset_sha256') != dataset_hash:
        raise ValueError('Teacher label dataset checksum mismatch')
    rows, annotations = dataset['entries'], labels['entries']
    ids = [row['id'] for row in rows]
    if not rows or not all(isinstance(identity, str) and identity for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError('Empty study or duplicate sample IDs')
    if [row['id'] for row in annotations] != ids:
        raise ValueError('Teacher sample alignment mismatch')
    groups = {}
    result = []
    for row, annotation in zip(rows, annotations):
        if (not isinstance(row.get('group_id'), str) or not row['group_id']
                or row.get('strategy') not in STRATEGIES
                or row.get('split') != 'evaluation-candidate'):
            raise ValueError('Invalid same-frame crop-study identity')
        strategies = groups.setdefault(row['group_id'], set())
        if row['strategy'] in strategies:
            raise ValueError('Duplicate group/strategy')
        strategies.add(row['strategy'])
        for key in ['group_id', 'strategy', 'file', 'sha256', 'scene_file', 'scene_sha256']:
            if annotation.get(key) != row[key]:
                raise ValueError('Teacher crop evidence identity mismatch')
        path = evidence_path(directory, row['file'], row['sha256'])
        evidence_path(directory, row['scene_file'], row['scene_sha256'])
        with Image.open(path) as image:
            if image.format != 'JPEG' or image.size != (384, 384) or image.mode != 'RGB':
                raise ValueError('Study requires exact 384x384 RGB JPEG evidence')
        consensus = annotation['teacher_consensus']
        if not isinstance(consensus.get('quality'), str) or not consensus['quality']:
            raise ValueError('Invalid teacher consensus quality')
        reference = consensus['material']
        if reference is not None and reference not in LABELS:
            raise ValueError('Unknown teacher material reference')
        result.append({**row, 'teacher_consensus': annotation['teacher_consensus'],
                       'teacher_reviews': annotation.get('teacher_reviews')})
    if any(strategies != set(STRATEGIES) for strategies in groups.values()):
        raise ValueError('Each group must have all four crop strategies')
    return dataset, labels, result


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def summarize(records):
    material = [row for row in records if row['reference_kind'] == 'material']
    referenced = [row for row in records if row['reference_kind'] != 'unresolved']
    accepted = [row for row in records if row['prediction']['label'] is not None]
    accepted_ref = [row for row in referenced if row['prediction']['label'] is not None]
    top1 = sum(row['prediction']['best_candidate'] == row['reference'] for row in material)
    top3 = sum(row['reference'] in [item['label'] for item in row['prediction']['candidates'][:3]]
               for row in material)
    accepted_correct = sum(row['prediction']['label'] == row['reference'] for row in accepted_ref)
    return {
        'examples': len(records), 'material_references': len(material),
        'rejection_references': sum(row['reference_kind'] == 'rejection' for row in records),
        'unresolved_references': sum(row['reference_kind'] == 'unresolved' for row in records),
        'consensus_quality_counts': dict(Counter(row['teacher_consensus']['quality'] for row in records)),
        'top1_correct': top1, 'top1_teacher_agreement': ratio(top1, len(material)),
        'top3_correct': top3, 'top3_teacher_agreement': ratio(top3, len(material)),
        'accepted': len(accepted), 'coverage': ratio(len(accepted), len(records)),
        'accepted_with_reference': len(accepted_ref),
        'accepted_reference_correct': accepted_correct,
        'accepted_reference_teacher_agreement': ratio(accepted_correct, len(accepted_ref)),
        'unresolved_accepted': sum(row['reference_kind'] == 'unresolved' for row in accepted),
        'rejection_accepted': sum(row['reference_kind'] == 'rejection' for row in accepted),
    }


def comparisons(records):
    groups = {}
    for row in records:
        variants = groups.setdefault(row['group_id'], {})
        if row['strategy'] in variants:
            raise ValueError('Duplicate comparison group/strategy')
        variants[row['strategy']] = row
    paired = {}
    for strategy in STRATEGIES[1:]:
        same, different, unresolved = [], [], []
        for group, variants in groups.items():
            baseline, variant = variants['stretch'], variants[strategy]
            if baseline['reference_kind'] != 'material' or variant['reference_kind'] != 'material':
                unresolved.append(group)
            elif baseline['reference'] != variant['reference']:
                different.append(group)
            else:
                same.append(group)
        base = [groups[group]['stretch'] for group in same]
        variant = [groups[group][strategy] for group in same]
        gained = lost = 0
        for before, after in zip(base, variant):
            was = before['prediction']['best_candidate'] == before['reference']
            now = after['prediction']['best_candidate'] == after['reference']
            gained += int(now and not was)
            lost += int(was and not now)
        paired[strategy] = {
            'eligible_same_material_groups': len(same), 'group_ids': same,
            'excluded_different_material_groups': len(different),
            'excluded_unresolved_or_rejection_groups': len(unresolved),
            'stretch': summarize(base), 'variant': summarize(variant),
            'top1_gained': gained, 'top1_lost': lost,
            'top1_agreement_delta': ratio(gained-lost, len(same)),
        }
    common = [group for group, variants in groups.items()
              if all(variants[strategy]['reference_kind'] == 'material' for strategy in STRATEGIES)
              and len({variants[strategy]['reference'] for strategy in STRATEGIES}) == 1]
    return {
        'paired_against_stretch': paired,
        'all_four_resolved_same_material': {
            'groups': len(common), 'group_ids': common,
            'strategies': {strategy: summarize([groups[group][strategy] for group in common])
                           for strategy in STRATEGIES},
        },
        'reference_resolution_by_group': {
            group: {strategy: {'reference': variants[strategy]['reference'],
                               'kind': variants[strategy]['reference_kind']}
                    for strategy in STRATEGIES}
            for group, variants in groups.items()},
    }


def evaluate_rows(directory, rows, classifier):
    records = []
    for row in rows:
        path = evidence_path(directory, row['file'], row['sha256'])
        with Image.open(path) as image:
            prediction = classifier.classify(np.array(image))
        reference = row['teacher_consensus']['material']
        kind = 'unresolved' if reference is None else 'rejection' if reference in REJECTIONS else 'material'
        records.append({**row, 'reference': reference, 'reference_kind': kind, 'prediction': prediction})
    return records


def write_exclusive(path, report):
    encoded = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(encoded)


def run(dataset_dir, model_dir, output, threads=2):
    if output.exists():
        raise ValueError('Use a new private report path')
    dataset, labels, rows = load_study(dataset_dir)
    classifier = MaterialClassifier(model_dir, device='CPU', threads=threads)
    records = evaluate_rows(dataset_dir, rows, classifier)
    report = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Descriptive agreement with two-pass weak VLM consensus on same-frame detector groups',
        'limitations': ['Teacher agreement is not material truth',
                        'Physical object identity, composition and novelty are unverified',
                        'Not a fresh object-disjoint holdout or a model-selection experiment'],
        'groups': len({row['group_id'] for row in records}),
        'dataset_sha256': digest(dataset_dir/'dataset.json'),
        'labels_sha256': digest(dataset_dir/'teacher-labels.json'),
        'capture_sha256': dataset.get('capture_sha256'),
        'teacher_protocol_sha256': labels.get('protocol_sha256'),
        'teacher_manifest_sha256': digest(dataset_dir/'teacher-manifest.json')
                                  if (dataset_dir/'teacher-manifest.json').is_file() else None,
        'model_manifest_sha256': digest(model_dir/'manifest.json'),
        'model_artifact_sha256': classifier.manifest['files'],
        'runtime': classifier.config,
        'by_strategy': {strategy: summarize([row for row in records if row['strategy'] == strategy])
                        for strategy in STRATEGIES},
        **comparisons(records), 'records': records,
    }
    write_exclusive(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Threads must be positive')
    result = run(args.dataset_dir, args.model_dir, args.output, args.threads)
    print(json.dumps({'groups': result['groups'], 'by_strategy': result['by_strategy'],
                      'output': str(args.output)}, indent=2))
