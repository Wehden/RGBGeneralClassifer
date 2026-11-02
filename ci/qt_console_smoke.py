"""Drive the actual X11 Qt/WebEngine console against a loopback synthetic service."""
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from rgb import desktop


def main():
    child = subprocess.Popen([sys.executable, 'ci/synthetic_service.py'], stdout=subprocess.PIPE, text=True)
    try:
        state = json.loads(child.stdout.readline())
        original = QApplication.exec
        failures, checks = [], []

        def controlled_exec(application):
            timer = QTimer()
            deadline = time.monotonic() + 35
            phase = 'dashboard'
            pending = False

            def poll():
                nonlocal phase, pending
                window = next((w for w in application.topLevelWidgets() if hasattr(w, 'page')), None)
                if time.monotonic() > deadline:
                    failures.append('Qt console did not load/render/close before deadline')
                    application.quit()
                    return
                if window is None or not window.loaded or pending:
                    return
                if phase == 'dashboard':
                    script = """JSON.stringify({
                      desktop:document.body.classList.contains('desktop-console'),
                      capture:!!document.querySelector('#pause-capture'),
                      comparison:!!document.querySelector('#comparison-content tbody tr')
                    })"""
                else:
                    script = """JSON.stringify({
                      desktop:document.body.classList.contains('desktop-console'),
                      records:!!document.querySelector('#rows tr')
                    })"""
                pending = True
                def inspected(value):
                    nonlocal phase, pending
                    pending = False
                    try:
                        result = json.loads(value or '{}')
                    except ValueError:
                        return
                    if not result or not all(result.values()):
                        return
                    checks.append(phase)
                    if phase == 'dashboard':
                        assert window.channel_actions['depth'].isEnabled()
                        assert window.channel_actions['pointcloud'].isEnabled()
                        phase = 'records'
                        window.navigate('/records')
                    else:
                        timer.stop()
                        window.close()  # Exercise real async WebEngine/profile teardown.
                window.page.runJavaScript(script, inspected)
            timer.timeout.connect(poll)
            timer.start(200)
            return original(application)

        QApplication.exec = controlled_exec
        sys.argv = ['rgb.desktop', '--attach', '--url', state['url'], '--software-rendering']
        assert desktop.main() == 0
        assert not failures, failures
        assert checks == ['dashboard', 'records'], checks
        with urlopen(state['url'] + '/healthz', timeout=3) as response:
            assert response.status == 200, 'Closing an attached console must leave its service running'
        print('PASS X11 Qt/WebEngine dashboard, records, depth menus and attached-service shutdown')
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        child.stdout.close()


if __name__ == '__main__':
    main()
