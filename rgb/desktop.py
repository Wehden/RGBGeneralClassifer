"""X11 Qt console: python -m rgb.desktop [console options] -- [runner options]."""
import argparse
import ctypes
import json
import os
import signal
import sys
import threading
from pathlib import Path

from rgb.desktop_service import ServiceController, local_url


def parser():
    result = argparse.ArgumentParser(
        description='X11 desktop console for the local Bolt RGB service',
        epilog='Runner options follow --, for example -- --no-vlm --state-dir /path/to/archive')
    result.add_argument('--attach', action='store_true', help='Only attach; never start a runner')
    result.add_argument('--url', help='Attach to a loopback service at this URL (implies --attach)')
    result.add_argument('--software-rendering', action='store_true',
                        help='Use software OpenGL for X forwarding or hosts without working graphics')
    result.add_argument('runner_options', nargs=argparse.REMAINDER)
    return result


def controller_for(args):
    options = args.runner_options
    if options[:1] == ['--']:
        options = options[1:]
    if args.url:
        if options:
            raise ValueError('Runner options cannot be applied when --url attaches to a service')
        return ServiceController(local_url(args.url))
    # Reuse the runner parser so defaults and environment precedence agree.
    from rgb.local import parser as runner_parser
    runtime = runner_parser().parse_args(options)
    if not 1 <= runtime.port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    url = f'http://127.0.0.1:{runtime.port}'
    if args.attach:
        if options and options != ['--port', str(runtime.port)]:
            raise ValueError('Use --url to attach; runtime settings belong to the running service')
        return ServiceController(url)
    if runtime.host not in {'127.0.0.1', 'localhost'}:
        raise ValueError('The desktop runner requires --host 127.0.0.1')
    return ServiceController(
        url, command=[sys.executable, '-m', 'rgb.local', *options,
                      '--host', '127.0.0.1', '--port', str(runtime.port)],
        state_dir=runtime.state_dir, cwd=str(Path(__file__).resolve().parent.parent))


def main():
    arguments = parser().parse_args()
    try:
        controller = controller_for(arguments)
    except ValueError as exc:
        raise SystemExit(str(exc))
    # xcb works on Xorg, Xwayland and forwarded X11 displays.
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    if os.environ.get('QT_QPA_PLATFORM') == 'xcb' and not os.environ.get('DISPLAY'):
        raise SystemExit('No X display. Set DISPLAY (or use ssh -X), then run rgb.desktop.')
    if arguments.software_rendering:
        os.environ['QT_OPENGL'] = 'software'
        # Keep Chromium's sandbox; only disable its GPU rendering.
        flags = os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
        os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = flags + ' --disable-gpu'
    if os.environ.get('QT_QPA_PLATFORM') == 'xcb':
        try:
            ctypes.CDLL('libxcb-cursor.so.0')
        except OSError:
            library = Path(sys.prefix) / 'lib/libxcb-cursor.so.0'
            if not library.exists():
                raise SystemExit('X11 dependency missing: libxcb-cursor0. Install it with your '
                                 'package manager, or run .venv/bin/python '
                                 'scripts/install-desktop-libs.py on Ubuntu.')
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    # OpenCV sets a Qt5 plugin/font path during import; this console uses Qt6.
    for key in ['QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR']:
        if '/cv2/qt/' in os.environ.get(key, ''):
            os.environ.pop(key)
    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QObject, QTimer, QUrl, Signal
        from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
        from PySide6.QtWidgets import (
            QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
            QStackedWidget, QToolBar, QVBoxLayout, QWidget,
        )
        from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        raise SystemExit('Desktop dependencies missing. Run .venv/bin/pip install '
                         '-r requirements-desktop.txt') from exc

    def same_origin(url):
        origin = QUrl(controller.url)
        return (url.scheme(), url.host(), url.port(443 if url.scheme() == 'https' else 80)) == (
            origin.scheme(), origin.host(), origin.port(443 if origin.scheme() == 'https' else 80))

    class Bridge(QObject):
        status = Signal(dict)
        stopped = Signal()

    class LocalPage(QWebEnginePage):
        def acceptNavigationRequest(self, url, kind, main_frame):
            if same_origin(url) or url.toString() == 'about:blank':
                return True
            if kind == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
                if url.scheme() in {'https', 'http'}:
                    QDesktopServices.openUrl(url)
            return False

    class Console(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Bolt RGB Console')
            self.resize(1440, 1000)
            self.loaded = False
            self.closing = False
            self.downloads = set()
            self.bridge = Bridge(self)
            self.bridge.status.connect(self.update_status)
            self.bridge.stopped.connect(self.finish_close)
            self.profile = QWebEngineProfile(self)  # Ephemeral browser state; application state is SQLite.
            self.profile.downloadRequested.connect(self.download_requested)
            self.view = QWebEngineView(self)
            self.page = LocalPage(self.profile, self.view)
            self.view.setPage(self.page)
            self.page.settings().setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
            self.page.fullScreenRequested.connect(self.fullscreen_requested)
            self.page.permissionRequested.connect(self.permission_requested)
            self.page.newWindowRequested.connect(self.new_window)
            self.view.loadFinished.connect(self.load_finished)
            self.stack = QStackedWidget()
            intro = QWidget()
            layout = QVBoxLayout(intro)
            self.message = QLabel('Connecting to the local camera service…')
            self.message.setWordWrap(True)
            self.message.setStyleSheet('font-size: 20px; padding: 40px')
            layout.addWidget(self.message)
            self.stack.addWidget(intro)
            self.stack.addWidget(self.view)
            self.setCentralWidget(self.stack)
            self.service_label = QLabel('Connecting…')
            self.statusBar().addWidget(self.service_label, 1)
            self.toolbar = QToolBar('Console')
            self.toolbar.setMovable(False)
            self.addToolBar(self.toolbar)
            file_menu = self.menuBar().addMenu('&File')
            view_menu = self.menuBar().addMenu('&View')
            self.action(file_menu, 'Dashboard', 'Ctrl+1', lambda: self.navigate('/'), toolbar=True)
            self.action(file_menu, 'Records', 'Ctrl+2', lambda: self.navigate('/records'), toolbar=True)
            self.toolbar.addSeparator()
            for index, channel in enumerate(['color', 'depth', 'ir', 'pointcloud'], 1):
                label={'color':'RGB','ir':'IR','pointcloud':'3D'}.get(channel,channel.title())
                self.action(view_menu, label, f'Alt+{index}',
                            lambda channel=channel: self.channel(channel), toolbar=True)
            self.action(view_menu, 'Reload', 'F5', self.view.reload)
            self.action(view_menu, 'Full screen', 'F11',
                        lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())
            self.action(view_menu, 'Zoom in', 'Ctrl++',
                        lambda: self.view.setZoomFactor(min(2, self.view.zoomFactor() + .1)))
            self.action(view_menu, 'Zoom out', 'Ctrl+-',
                        lambda: self.view.setZoomFactor(max(.5, self.view.zoomFactor() - .1)))
            self.action(view_menu, 'Reset zoom', 'Ctrl+0', lambda: self.view.setZoomFactor(1))
            downloads_menu = file_menu.addMenu('Download')
            for title, path, name in [
                ('Color snapshot', '/api/camera/color/snapshot.jpg', 'bolt-color.jpg'),
                ('Depth preview', '/api/camera/depth/snapshot.jpg', 'bolt-depth.jpg'),
                ('IR preview', '/api/camera/ir/snapshot.jpg', 'bolt-ir.jpg'),
                ('Raw depth', '/api/camera/depth/raw.png', 'bolt-depth-uint16.png'),
                ('Raw IR', '/api/camera/ir/raw.png', 'bolt-ir-uint16.png'),
                ('Sensor frame bundle', '/api/camera/frame', 'bolt-frame.npz'),
                ('3D point cloud frame', '/api/camera/pointcloud', 'bolt-pointcloud.bin'),
                ('Parquet archive', '/api/review/export/parquet', 'rgb-review.zip'),
                ('Arrow archive', '/api/review/export/arrow', 'rgb-review.zip'),
            ]:
                self.action(downloads_menu, title, None,
                            lambda path=path, name=name: self.page.download(
                                QUrl(controller.url + path), name))
            self.action(file_menu, 'Open in browser', 'Ctrl+L',
                        lambda: QDesktopServices.openUrl(QUrl(controller.url)))
            self.log_action = self.action(file_menu, 'Open runner log', 'Ctrl+Shift+L',
                                         self.open_log)
            self.log_action.setEnabled(False)
            self.action(view_menu, 'Pause/resume capture', 'Ctrl+P',
                        lambda: self.page.runJavaScript("document.getElementById('pause-capture')?.click()"))
            file_menu.addSeparator()
            self.action(file_menu, 'Quit', 'Ctrl+Q', self.close)
            threading.Thread(target=controller.watch, args=(self.bridge.status.emit,),
                             name='desktop-status', daemon=True).start()

        def action(self, menu, label, shortcut, callback, *, toolbar=False):
            action = QAction(label, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda _checked=False: callback())
            menu.addAction(action)
            if toolbar:
                self.toolbar.addAction(action)
            return action

        def navigate(self, path):
            self.view.load(QUrl(controller.url + path))

        def channel(self, name):
            script = """(() => {
              const select = document.getElementById('camera-channel');
              if (!select) return false;
              select.value = %s;
              select.dispatchEvent(new Event('change'));
              return true;
            })()""" % json.dumps(name)
            def selected(ok):
                if not ok:
                    self.service_label.setText('Choose Dashboard first to change the live camera view.')
            self.page.runJavaScript(script, selected)

        def update_status(self, status):
            if self.closing:
                return
            self.log_action.setEnabled(controller.owned)
            if status['state'] != 'ready':
                text = status['message'] + ' · ' + controller.url
                self.service_label.setText(text)
                self.message.setText(text)
                return
            health, vlm = status['data']['health'], status['data']['vlm']
            camera = health['camera']
            lifecycle = 'Console started service; closing stops it' if status['owned'] else 'Attached; service stays running'
            model = f"{vlm.get('selected_provider', 'bedrock')} / {vlm.get('selected_model', 'unknown')}"
            mode = 'disabled' if not vlm['enabled'] else 'paused' if vlm.get('paused') else 'active'
            text = (f"{lifecycle} · Camera {camera.get('state', 'unknown')} "
                    f"({camera.get('capture_fps', 0):.1f} FPS) · {model} {mode} · {controller.url}")
            if camera.get('error'):
                text += ' · ' + camera['error']
            self.service_label.setText(text)
            if not self.loaded:
                self.loaded = True
                self.navigate('/')

        def load_finished(self, ok):
            if self.closing:
                return
            if ok:
                self.page.runJavaScript("document.body.classList.add('desktop-console')")
                self.stack.setCurrentWidget(self.view)
            elif not self.closing:
                self.service_label.setText('Page failed to load. Use View → Reload after the service recovers.')

        def open_log(self):
            if controller.log_path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(controller.log_path.resolve())))

        def permission_requested(self, permission):
            # Location remains opt in through the existing explicit UI action.
            if same_origin(permission.origin()) and permission.permissionType().name == 'Geolocation':
                answer = QMessageBox.question(self, 'Camera location',
                    'Allow this console to obtain the desktop location for the selected camera?')
                if answer == QMessageBox.StandardButton.Yes:
                    permission.grant()
                    return
            permission.deny()

        def new_window(self, request):
            if same_origin(request.requestedUrl()):
                request.openIn(self.page)
            elif request.isUserInitiated() and request.requestedUrl().scheme() in {'http', 'https'}:
                QDesktopServices.openUrl(request.requestedUrl())

        def fullscreen_requested(self, request):
            request.accept()
            self.showFullScreen() if request.toggleOn() else self.showNormal()

        def download_requested(self, download):
            name = download.suggestedFileName().replace('\\', '/').split('/')[-1] or 'download'
            home = Path.home() / 'Downloads'
            target, _ = QFileDialog.getSaveFileName(self, 'Save camera or review data', str(home / name))
            if not target:
                download.cancel()
                return
            destination = Path(target)
            download.setDownloadDirectory(str(destination.parent))
            download.setDownloadFileName(destination.name)
            self.downloads.add(download)
            def changed():
                state = download.state()
                if state.name == 'DownloadCompleted':
                    self.statusBar().showMessage('Saved ' + str(destination), 10000)
                elif state.name == 'DownloadInterrupted':
                    self.statusBar().showMessage('Download failed: ' + download.interruptReasonString(), 15000)
                if download.isFinished():
                    self.downloads.discard(download)
            download.stateChanged.connect(changed)
            download.accept()

        def closeEvent(self, event):
            event.ignore()
            if self.closing:
                return
            self.closing = True
            controller.request_stop()
            self.setEnabled(False)
            self.service_label.setText('Closing console and stopping its service…' if controller.owned
                                       else 'Closing console; attached service stays running…')
            self.page.setUrl(QUrl('about:blank'))  # Release this console's MJPEG viewer immediately.
            for download in list(self.downloads):
                download.cancel()
            def shutdown():
                try:
                    controller.close()
                finally:
                    self.bridge.stopped.emit()
            threading.Thread(target=shutdown, name='desktop-shutdown', daemon=True).start()

        def finish_close(self):
            self.hide()
            self.page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.profile.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            application.quit()

    application = QApplication([sys.argv[0]])
    application.setApplicationName('Bolt RGB Console')
    application.setOrganizationName('RGBGeneralClassifier')
    application.setQuitOnLastWindowClosed(False)
    window = Console()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    pulse = QTimer()
    pulse.timeout.connect(lambda: None)  # Let Python process POSIX signals while Qt is idle.
    pulse.start(250)
    window.show()
    try:
        return application.exec()
    finally:
        controller.close()


if __name__ == '__main__':
    raise SystemExit(main())
