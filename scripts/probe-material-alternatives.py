#!/usr/bin/env python3
"""Validation-only NumPy probes on immutable, normalized material embeddings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


REJECTIONS = {'human skin', 'food', 'unknown'}
LABELS = ['plastic', 'metal', 'glass', 'wood', 'paper', 'cardboard', 'fabric', 'rubber',
          'ceramic', 'leather', 'stone', 'liquid', 'human skin', 'food', 'unknown']
ALLOCATIONS = [
    ('legacy', None),
    ('public_only', {'MINC': 1., 'camera_material': 0., 'camera_rejection': 0.}),
    ('camera20', {'MINC': .7, 'camera_material': .2, 'camera_rejection': .1}),
    ('camera40', {'MINC': .5, 'camera_material': .4, 'camera_rejection': .1}),
    ('camera40_reject20', {'MINC': .4, 'camera_material': .4, 'camera_rejection': .2}),
    ('camera_only', {'MINC': 0., 'camera_material': .75, 'camera_rejection': .25}),
]


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cohort(row):
    if row['source_kind'] == 'MINC':
        return 'MINC'
    return 'camera_rejection' if row['material'] in REJECTIONS else 'camera_material'


def training_weights(rows, allocation=None):
    """Class-balance within each domain; all statistics use training rows only."""
    ids = [i for i, row in enumerate(rows) if row['split'] == 'train' and row['target'] >= 0
           and row['weight'] > 0]
    if not ids:
        raise ValueError('No labeled training rows')
    counts = Counter(rows[i]['target'] for i in ids)
    if allocation is None:
        weights = np.array([rows[i]['weight']/np.sqrt(counts[rows[i]['target']]) for i in ids])
    else:
        counts = Counter((cohort(rows[i]), rows[i]['target']) for i in ids)
        weights = np.array([rows[i]['weight']/counts[(cohort(rows[i]), rows[i]['target'])] for i in ids])
        for kind, mass in allocation.items():
            selected = np.array([cohort(rows[i]) == kind for i in ids])
            total = weights[selected].sum()
            if mass > 0 and total == 0:
                raise ValueError('Requested empty training cohort')
            weights[selected] *= mass/total if total else 0
    weights /= weights.sum()
    return np.array(ids), weights


def classify(train, labels, weights, validation, method, parameter):
    supported = np.unique(labels[weights > 0])
    if method == 'prototype':
        scores = np.full((len(validation), len(LABELS)), -1e9)
        for target in supported:
            selected = labels == target
            center = np.average(train[selected], axis=0, weights=weights[selected])
            center /= max(np.linalg.norm(center), 1e-12)
            scores[:, target] = validation @ center
    elif method == 'ridge':
        # The intercept is fitted and regularized; weights have unit total mass.
        design = np.column_stack([train, np.ones(len(train))])
        targets = np.eye(len(LABELS))[labels]
        covariance = design.T @ (design*weights[:, None])
        coefficient = np.linalg.solve(covariance+parameter*np.eye(design.shape[1]),
                                      design.T @ (targets*weights[:, None]))
        scores = np.column_stack([validation, np.ones(len(validation))]) @ coefficient
        scores[:, ~np.isin(np.arange(len(LABELS)), supported)] = -1e9
    elif method == 'knn':
        active = weights > 0
        similarity = validation @ train[active].T
        k, temperature = parameter
        k = min(k, similarity.shape[1])
        nearest = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
        selected = np.take_along_axis(similarity, nearest, axis=1)
        influence = np.exp((selected-selected.max(axis=1, keepdims=True))/temperature)*weights[active][nearest]
        scores = np.zeros((len(validation), len(LABELS)))
        for index in range(len(validation)):
            np.add.at(scores[index], labels[active][nearest[index]], influence[index])
        scores[:, ~np.isin(np.arange(len(LABELS)), supported)] = -1e9
    else:
        raise ValueError('Unknown classifier')
    return scores


def validation_metrics(scores, rows):
    if any(row['split'] != 'validation' for row in rows):
        raise ValueError('Probe evaluation permits validation rows only')
    order = np.argsort(-scores, axis=1, kind='stable')
    result = {}
    for kind in ['MINC', 'camera']:
        ids = [i for i, row in enumerate(rows) if row['source_kind'] == kind and row['target'] >= 0
               and row['material'] not in REJECTIONS]
        classes = sorted({rows[i]['target'] for i in ids})
        by_class = {}
        for target in classes:
            selected = [i for i in ids if rows[i]['target'] == target]
            by_class[LABELS[target]] = {'correct': sum(int(order[i, 0] == target) for i in selected),
                                      'references': len(selected)}
        groups = {}
        for i in ids:
            group = rows[i].get('group_id', rows[i]['id'])
            groups.setdefault(group, []).append(int(order[i, 0] == rows[i]['target']))
        result[kind] = {
            'references': len(ids),
            'correct': sum(int(order[i, 0] == rows[i]['target']) for i in ids),
            'top1': np.mean([order[i, 0] == rows[i]['target'] for i in ids]).item() if ids else None,
            'top3': np.mean([rows[i]['target'] in order[i, :3] for i in ids]).item() if ids else None,
            'class_macro': float(np.mean([v['correct']/v['references'] for v in by_class.values()])) if classes else None,
            'group_macro': float(np.mean([np.mean(v) for v in groups.values()])) if groups else None,
            'per_class': by_class,
        }
        rejects = [i for i, row in enumerate(rows) if row['source_kind'] == kind and row['target'] >= 0
                   and row['material'] in REJECTIONS]
        result[kind]['rejection_references'] = len(rejects)
        result[kind]['rejection_top1_correct'] = sum(int(order[i, 0] == rows[i]['target']) for i in rejects)
    public = result['MINC']['top1'] or 0
    camera = result['camera']['top1'] or 0
    result['selection_objective'] = .7*public+.3*camera if result['camera']['references'] >= 10 else public
    return result


def run(features, ids, rows, baselines):
    if features.ndim != 2 or features.shape[1] < 1 or features.shape[0] != len(rows):
        raise ValueError('Expected a nonempty-width feature matrix aligned with rows')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample identity')
    splits = {row['split'] for row in rows}
    if not splits <= {'train', 'validation', 'test'} or not {'train', 'validation'} <= splits:
        raise ValueError('Expected train and validation splits; only train/validation/test are supported')
    if ids.tolist() != [row['id'] for row in rows]:
        raise ValueError('Feature/sample alignment mismatch')
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
        raise ValueError('Expected finite normalized embeddings')
    validation_ids = [i for i, row in enumerate(rows) if row['split'] == 'validation']
    validation_rows = [rows[i] for i in validation_ids]
    held = features[validation_ids].astype(np.float64)
    results = []
    for name, allocation in ALLOCATIONS:
        train_ids, weights = training_weights(rows, allocation)
        train = features[train_ids].astype(np.float64)
        targets = np.array([rows[i]['target'] for i in train_ids])
        mass = {kind: float(weights[[cohort(rows[i]) == kind for i in train_ids]].sum())
                for kind in ['MINC', 'camera_material', 'camera_rejection']}
        configurations = [('prototype', None)]
        configurations += [('ridge', value) for value in [.0001, .001, .01, .1]]
        configurations += [('knn', (k, t)) for k in [1, 5, 15, 30] for t in [.02, .1]]
        for method, parameter in configurations:
            scores = classify(train, targets, weights, held, method, parameter)
            results.append({'allocation': name, 'method': method, 'parameter': parameter,
                            'training_mass': mass, 'validation': validation_metrics(scores, validation_rows)})
    baseline_metrics = {name: validation_metrics(values[validation_ids], validation_rows)
                        for name, values in baselines.items()}
    best = max(results, key=lambda row: row['validation']['selection_objective'])
    return {'scope': 'validation-only exploratory selection; camera references are weak VLM labels',
            'training_rows': len(training_weights(rows)[0]), 'validation_rows': len(validation_ids),
            'historical_test_evaluated': False, 'baselines': baseline_metrics,
            'best_by_validation': best, 'candidates': results,
            'camera_domain_limit': 'Validation target materials cover only fabric and plastic; fresh-object confirmation required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--feature-file', type=Path, help='Alternate normalized frozen embeddings')
    parser.add_argument('--rows-file', type=Path, help='Fixed train/validation sample manifest, without historical test rows')
    parser.add_argument('--skip-baselines', action='store_true', help='Do not load original-head logits for independent encoder probes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.run_directory
    feature_file = args.feature_file or source/'features.npz'
    rows_file = args.rows_file or source/'evaluation-rows.json'
    rows = json.loads(rows_file.read_text())
    if args.rows_file and any(row['split'] not in {'train', 'validation'} for row in rows):
        raise ValueError('Explicit rows-file permits train/validation rows only')
    with np.load(feature_file, allow_pickle=False) as data:
        features, ids = data['features'].copy(), data['ids'].copy()
    baselines = {}
    input_files = {'features.npz': feature_file, 'evaluation-rows.json': rows_file}
    if not args.skip_baselines:
        logits_file = source/'evaluation-logits.npz'
        with np.load(logits_file, allow_pickle=False) as data:
            baseline_ids = data['ids'].tolist()
            if len(set(baseline_ids)) != len(baseline_ids):
                raise ValueError('Duplicate baseline sample identity')
            if not args.feature_file and not args.rows_file and baseline_ids != ids.tolist():
                raise ValueError('Baseline/sample alignment mismatch')
            indexes = {identity: i for i, identity in enumerate(baseline_ids)}
            if any(identity not in indexes for identity in ids.tolist()):
                raise ValueError('Baseline/sample alignment mismatch')
            selected = [indexes[identity] for identity in ids.tolist()]
            baselines = {'frozen_head': data['frozen'][selected].copy(),
                         'fine_tuned_head': data['tuned'][selected].copy()}
        input_files['evaluation-logits.npz'] = logits_file
    result = run(features, ids, rows, baselines)
    result['inputs'] = {name: checksum(path) for name, path in input_files.items()}
    result['input_paths'] = {name: str(path.resolve()) for name, path in input_files.items()}
    result['script_sha256'] = checksum(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'best_by_validation': result['best_by_validation'], 'baselines': result['baselines']}))


if __name__ == '__main__':
    main()
