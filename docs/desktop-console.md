# X11 desktop console

The Qt console uses the same dashboard and Records pages as the browser.
Color, registered depth, native IR, object/material modes, thresholds, model
selection, pause/resume, archive capture, evaluations and evidence all use
the existing local API and SQLite archive. Qt supplies the desktop window,
menus, shortcuts, full-screen mode, connection status and native save dialogs.
The console has a compact layout so the camera remains visible below its
controls. It does not acquire the camera or run another model itself.

## Install and launch

```sh
.venv/bin/pip install -r requirements-desktop.txt
.venv/bin/python -m rgb.desktop
```

Xorg, Xwayland and forwarded X11 displays use Qt's `xcb` platform by default.
`DISPLAY` and the normal X authority must allow access. This desktop already
has a working Xwayland display; no display configuration was changed.

Qt also needs the system XCB cursor library. On Ubuntu, install
`libxcb-cursor0` with the package manager. If system installation is unavailable,
this helper downloads the Ubuntu package matching local APT metadata,
verifies its SHA-256 and extracts that library into `.venv/lib`:

```sh
.venv/bin/python scripts/install-desktop-libs.py
```

The helper has been run on this machine. It changes only the virtualenv.
Other XCB dependencies must already be supplied by the desktop environment.
OpenCV's Qt5 plugin path is excluded from the console's Qt6 setup; capture
continues in the service process.

To add **Bolt RGB Console** to the desktop application menu and the `bolt-rgb`
command to `~/.local/bin`:

```sh
.venv/bin/python scripts/install-desktop-launcher.py
```

Launch from any directory with `bolt-rgb`. Arguments are forwarded unchanged,
for example `bolt-rgb --attach` or `bolt-rgb -- --no-vlm`. The helper selects
the checkout and its `.venv` automatically; relative runner paths resolve
from the checkout.

`~/.local/bin` is already on this machine's PATH. On another machine, add it
to your shell PATH. The menu entry and helper use this checkout and its `.venv`.
Reinstall after moving the checkout, removing the old symlink first. Remove
`~/.local/share/applications/bolt-rgb.desktop` and `~/.local/bin/bolt-rgb`
to uninstall the launchers.

## Service ownership

By default the console probes `http://127.0.0.1:8080`. If a local Bolt RGB
service is already running, it attaches and leaves that service running when
the window closes. Its status bar states the ownership explicitly. Opening
a console and browser together uses one capture owner.

If no service responds, the console starts `python -m rgb.local`. Closing
the window stops that child gracefully, allowing up to 12 seconds for its
10-second shutdown deadline. A child that still does not exit is terminated.
Logs are appended to `<state-dir>/desktop-service.log`. A responding listener
that fails the local-service health contract is shown as an error, rather
than treated as an invitation to launch another service.

The default runner configuration remains profile `default`, region
`us-west-2`, Nova 2 Lite, and archive `~/.local/share/bolt-rgb`.
Normal live inference makes paid calls through the selected VLM provider.
Bedrock is the default; [OpenRouter setup](local-operations.md#vlm-providers)
uses `OPENROUTER_API_KEY` in the service environment. Pass runtime options after
`--` to configure a newly started service:

```sh
# Camera and archive only; no cloud requests:
.venv/bin/python -m rgb.desktop -- --no-vlm

# Start paused with a specific private archive:
.venv/bin/python -m rgb.desktop -- --paused --state-dir /path/to/archive

# Configure a newly started service on another port:
.venv/bin/python -m rgb.desktop -- --port 8083 --aws-profile default

# Attach only; never launch or stop the service:
.venv/bin/python -m rgb.desktop --attach
.venv/bin/python -m rgb.desktop --url http://127.0.0.1:8083

# Software rendering for graphics compatibility or SSH X forwarding:
.venv/bin/python -m rgb.desktop --software-rendering --attach
```

Runtime options do not override an already running service. Use dashboard
settings for saved controls, or restart the service to change its startup
configuration. `--url` implies attachment and cannot be combined with runtime
options. Console connections accept only loopback origins. An SSH local
port forward can expose a remote service at a local origin without changing
the service's bind.

The window remains responsive during startup, API outages and shutdown;
network probes and process waits run outside the GUI thread. An attached
service that appears later is loaded automatically. A failed page load can
be retried with F5. A service outage does not automatically navigate away
from an open review draft.

## Menus and shortcuts

| Action | Shortcut |
| --- | --- |
| Dashboard / Records | Ctrl+1 / Ctrl+2 |
| RGB / Depth / IR / 3D on the dashboard | Alt+1 / Alt+2 / Alt+3 / Alt+4 |
| Pause/resume automatic capture on the dashboard | Ctrl+P |
| Reload current page | F5 |
| Full screen | F11 |
| Zoom in / out / reset | Ctrl++ / Ctrl+- / Ctrl+0 |
| Open dashboard in browser | Ctrl+L |
| Open console-owned runner log | Ctrl+Shift+L |
| Quit | Ctrl+Q |

File → Download saves snapshots, original uint16 depth/IR, the sensor NPZ,
the current 3D point-cloud frame, and Arrow/Parquet archive ZIPs. Existing page download links, including
per-observation evidence, also use a native save dialog. Canceled dialogs
cancel downloads. Closing cancels this console's downloads and releases its
MJPEG viewer.

External HTTP(S) links open in the default browser only for user navigation.
Desktop location is requested only after the existing location button is
used and a Qt permission dialog is accepted. The console does not request
browser camera/microphone access, and Chromium's sandbox remains enabled.

## Verifiable desktop claims

| ID | Claim | Evidence/check |
| --- | --- | --- |
| D01 | Runs in an X11 Qt window with all three live views. | Actual Xwayland/xcb run; snapshots of color/depth/IR and native toolbar switching. |
| D02 | Uses the existing local service, with no second sensor owner. | Real attached run; lifecycle tests require no child spawn for an existing service. |
| D03 | Closing an attached console leaves the service running; closing an owning console stops its child. | Automated subprocess ownership/close-race tests and actual camera service lifecycle checks. |
| D04 | Existing review/control behavior remains shared. | Same HTML/JS/API; browser contract suite and desktop records/navigation/download checks. |
| D05 | Native downloads preserve sensor and typed archive data. | Desktop NPZ dtype checks, raw uint16 PNG decode, archive ZIP table check. |
| D06 | Startup/network/shutdown waits do not block the GUI; errors are visible. | Worker lifecycle tests, actual asynchronous startup/close, runner failure reported without respawn. |
| D07 | Default runner and console connections use loopback; an unexpected listener is not replaced. | URL rejection/probe tests, default bind and actual listener inspection. |

See [dated verification](local-verification.md) for measured results and limits.
