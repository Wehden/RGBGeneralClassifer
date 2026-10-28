#!/usr/bin/env python3
"""Join offline material predictions to blinded two-pass references by exact crop."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rgb.material_agreement import compare, reference, summarize


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(review_directory, receipts):
    cases_path = review_directory/'cases.json'
    cases = json.loads(cases_path.read_text())
    lookup = {case['case_id']:case for case in cases}
    if len(lookup) != len(cases):
        raise ValueError('Duplicate reference crop identities')
    for case in cases:
        if digest(case['crop_file']) != case['crop_sha256']:
            raise ValueError('Reference crop changed')
    review_summary = json.loads((review_directory/'summary.json').read_text())
    models = {}
    for path in receipts:
        result = json.loads(path.read_text())
        if result.get('state') != 'passed':
            raise ValueError('Material prediction run did not pass')
        rows = result['readings']
        for row in rows:
            case = lookup.get(row['case_id'])
            if case is None or row['sha256'] != case['crop_sha256']:
                raise ValueError('Material prediction/reference crop mismatch')
        metrics = compare(cases, rows)
        confusion = defaultdict(Counter)
        for row in rows:
            ref = reference(lookup[row['case_id']]['reviews'])
            if ref['dominant'] is not None and not ref['out_of_scope']:
                confusion[ref['dominant']][row['best_candidate']] += 1
        config = result['config']
        models[result['model']] = {
            **metrics, 'receipt_sha256':digest(path),
            'config':{key:config[key] for key in
                      ['model', 'revision', 'manifest_sha256', 'artifact_sha256', 'input_size']},
            'top1_dominant_confusion':{key:dict(value) for key, value in sorted(confusion.items())}}
    refs = [reference(case['reviews']) for case in cases]
    dominant_distribution = Counter(ref['dominant'] for ref in refs if ref['dominant'] is not None)
    return {'reviewer':summarize(cases), 'reviewer_run':review_summary,
            'reference_cases_sha256':digest(cases_path),
            'dominant_reference_distribution':dict(sorted(dominant_distribution.items())),
            'models':models,
            'interpretation':'Blinded visual-material agreement, not chemical composition or human-ground-truth accuracy',
            'dominant_metric':'Top choice equals stable two-pass target dominant material / stable target dominant references',
            'supported_metric':'Diagnostic: top choice belongs to materials supported in both passes / crops with supported target materials',
            'accepted_metric':'Agreement conditional on accepted predictions with references; undefined if none'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate(args.reviews, args.receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.temporary')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps({name:{key:value for key,value in result.items() if key in
                     ['crops', 'dominant_references', 'top1_dominant_matches', 'top1_dominant_agreement',
                      'supported_references', 'top1_supported_agreement', 'acceptance_coverage',
                      'accepted_dominant_agreement']} for name,result in summary['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
