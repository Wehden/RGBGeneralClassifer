"""Enforce branch-inclusive coverage floors for the full unit suite."""
import argparse
import json
from pathlib import Path


def check(report, budgets):
    failures = []
    summaries = {'TOTAL': report['totals']}
    summaries.update({name: item['summary'] for name, item in report['files'].items()})
    for name, floor in budgets.items():
        summary = summaries.get(name)
        actual = summary['percent_covered'] if summary else None
        if actual is None or actual + 1e-6 < floor:
            failures.append(f'{name}: {actual if actual is not None else "missing"} < {floor}%')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--budgets', type=Path, default=Path('ci/coverage-budgets.json'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    failures = check(report, json.loads(args.budgets.read_text()))
    print(f"Statement + branch coverage: {report['totals']['percent_covered']:.2f}%")
    if failures:
        raise SystemExit('\n'.join(failures))
    print('All coverage floors passed.')


if __name__ == '__main__':
    main()
