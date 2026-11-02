"""The gate rejects missing modules, regressions and branch-only losses."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'coverage_gate', Path(__file__).resolve().parents[1] / 'scripts/check-coverage.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_coverage_gate_checks_aggregate_and_individual_modules():
    report = {'totals': {'percent_covered': 80},
              'files': {'rgb/web.py': {'summary': {'percent_covered': 70}}}}
    assert gate.check(report, {'TOTAL': 80, 'rgb/web.py': 70}) == []
    assert len(gate.check(report, {'TOTAL': 81, 'rgb/web.py': 71, 'missing.py': 0})) == 3


def test_branch_coverage_is_part_of_the_gate():
    # The coverage JSON percentage already includes missed branches.
    report = {'totals': {'percent_covered': 60, 'covered_lines': 10,
                         'num_statements': 10, 'num_branches': 10, 'covered_branches': 2},
              'files': {}}
    assert gate.check(report, {'TOTAL': 80}) == ['TOTAL: 60 < 80%']
