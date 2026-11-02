"""Deployment qualification must fail when optional dependencies are missing."""
import pytest


def pytest_sessionfinish(session, exitstatus):
    terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
    skipped = terminalreporter.stats.get('skipped', []) if terminalreporter else []
    if skipped:
        terminalreporter.write_sep('!', 'Qualification tests may not skip')
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
