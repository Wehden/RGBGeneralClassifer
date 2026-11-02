"""The optional-runtime lanes must not silently turn missing coverage green."""
from pathlib import Path
import subprocess
import sys


def test_qualification_plugin_fails_a_skipped_test(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'conftest.py').write_text((root / 'ci/conftest.py').read_text())
    test = tmp_path / 'test_skip.py'
    test.write_text('import pytest\n\ndef test_missing_dependency():\n    pytest.skip("synthetic absent runtime")\n')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', str(test)],
        cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Qualification tests may not skip' in result.stdout
