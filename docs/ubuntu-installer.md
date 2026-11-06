# Local Ubuntu installer

The `bolt-rgb` Debian package targets **Ubuntu 24.04 LTS on Intel/AMD
64-bit (amd64) and ARM64 (arm64/aarch64), with Python 3.12**. It installs RGB Console, its
local web service, the pinned nano fallback detector, a desktop menu entry,
and USB permission rules for the Femto Bolt and RealSense D415. The runtime uses the existing
`requirements-desktop.txt` pins and an isolated, per-user virtualenv.
32-bit ARM and other Ubuntu releases are not supported by this installer.

## Build on the current system

From this checkout:

```sh
python3 scripts/build-ubuntu-package.py
```

This detects the native build host and creates `dist/bolt-rgb_0.YYYYMMDD.1_amd64.deb`
or `dist/bolt-rgb_0.YYYYMMDD.1_arm64.deb`, plus a SHA-256 receipt.
Use `--architecture arm64` or `--architecture amd64` to build a small package
for another architecture. The control metadata and packaged target receipt
identify the selected architecture; this is not just a filename change.
The small package downloads Python dependencies during user setup on the
destination. Building needs `dpkg-deb`; it does not require sudo.

For a move where Python dependencies should travel with the installer, build
on Ubuntu 24.04 with Python 3.12 and pip, on the **same architecture as the
destination** (ventunoq for ARM64, an Intel/AMD host for amd64):

```sh
.venv/bin/python scripts/build-ubuntu-package.py --bundle-wheels
```

This includes the complete resolved wheel set, including the Orbbec and RealSense SDK bindings,
ONNX Runtime, OpenVINO, and Qt. Setup verifies each wheel's SHA-256 and uses
only those files. Cross-architecture wheel bundling is rejected. The bundled package is much larger. Build either variant
with an explicit version for upgrades, for example `--version 0.20261105.2`.
Each build replaces the same-version output file.

Only application code, UI assets, requirements, licenses, and the pinned
fallback model enter the package. Archives, settings, AWS/OpenRouter
credentials, training data, model caches, and the checkout's virtualenvs
are excluded.

## Install and run on the destination

Copy the `.deb` and its `.deb.sha256` file to the destination and run from
their directory, substituting the filename printed by the build. Use `arm64`
on ventunoq; `dpkg --print-architecture` reports the destination architecture:

```sh
sha256sum -c bolt-rgb_0.20261106.1_arm64.deb.sha256
sudo apt install ./bolt-rgb_0.20261106.1_arm64.deb
bolt-rgb-setup
bolt-rgb
```

APT installs Python's venv support, libusb, and the Qt/X11/WebEngine system
libraries. Run setup and the console as the normal desktop user, without
sudo. Setup can be run again safely; it resumes incomplete dependency
installation and reuses a completed environment. Runtime cache keys include
the architecture, packaged target receipt, and native Qt validation version.
Setup imports WebEngine, not just the PySide6 package; ARM64 WebEngine
requires the declared codec, font, compression, and XCB libraries, including
`libsnappy1v5` and `libminizip1t64`. Setup refuses a mismatched
architecture, Python version, or Ubuntu release before invoking pip.
Package staging normalizes directory/file permissions even under `umask 077`;
private user runtime data still uses restrictive permissions.
For a bundled package, `bolt-rgb-setup --offline` explicitly requires offline
Python setup. APT still needs Ubuntu packages from its repositories or a
previously populated cache: bundling Python wheels does not make installation
on an unprepared machine entirely offline.

Reconnect the camera after installation. The permission rule applies only
to the Femto Bolt (`2bc5:066b`) and RealSense D415 (`8086:0ad3`), grants the active local desktop user access,
and gives the `video` group read/write access. For a headless or SSH-only
user, add that account to `video`, then log out and in:

```sh
sudo usermod -aG video "$USER"
```

Launch **RGB Console** from the desktop menu, or use `bolt-rgb` /
`rgb-console` from a terminal. For capture without cloud inference:

```sh
bolt-rgb -- --no-vlm
# Alternatively, run just the browser/API service:
bolt-rgb-service --no-vlm
```

For the RealSense camera, close the console/service that owns the current
camera, then launch:

```sh
bolt-rgb -- --camera-backend realsense --no-vlm
# Alternatively:
bolt-rgb-service --camera-backend realsense --no-vlm
```

The RealSense binding is included in both installer variants. Existing
librealsense kernel drivers and firmware are managed by the destination system.
Use the SDK-reported `--camera-serial` for device selection; the USB/ASIC serial
can differ. Close RealSense Viewer before starting console capture.
The D415 supplies RGB, native uint16 depth, native uint8 left IR, and depth
registered to the color camera for the 3D viewer.
See [RealSense operation](local-operations.md#realsense-cameras) for calibration
and measurement limitations.

The service is available at `http://127.0.0.1:8080`. The packaged service
command always binds to loopback. The desktop uses the existing attach/start
lifecycle: it stops only a service it starts. Installation and setup do not
start camera capture, change archive settings, or invoke cloud models.
Normal launch retains the application's existing capture and VLM defaults;
use `--no-vlm` for the first camera check if cloud credentials are not ready.

The Qt console needs an X11 or Xwayland display. GPU drivers remain the
destination system's responsibility; `bolt-rgb --software-rendering` is
available for software OpenGL rendering.

## Carry over the existing installation

Stop the owning console/service on the old system before copying the capture
archive. Copy the entire archive, including its SQLite files and `assets`
directory together. The default paths are:

| Content | Path |
| --- | --- |
| Capture archive and saved settings | `~/.local/share/bolt-rgb/` |
| Larger detector and material models | `~/.cache/bolt-rgb/models/` |
| Installed package code | `/opt/bolt-rgb/` |
| New per-user Python environments | `~/.local/share/bolt-rgb-runtime/` |

For example, with empty destination directories, run on the old system:

```sh
ssh USER@NEW-HOST 'mkdir -p ~/.local/share/bolt-rgb ~/.cache/bolt-rgb/models'
rsync -a ~/.local/share/bolt-rgb/ USER@NEW-HOST:.local/share/bolt-rgb/
rsync -a --copy-links ~/.cache/bolt-rgb/models/ USER@NEW-HOST:.cache/bolt-rgb/models/
```

`--copy-links` copies the actual model directories behind activation links,
so an absolute link to the old home directory does not break on the new host.
If the old service used `RGB_STATE_DIR` or `--state-dir`, migrate that directory
instead and pass the equivalent path to the new service.
Do not copy the old `.venv`; setup installs a fresh environment on the new host.
Credentials are configured separately for the destination user. The package
neither reads nor transports them. A model cache copy preserves the current
small detector and learned material model; omitting it uses the bundled nano
fallback and leaves optional local material inference unavailable.
See [private model releases](model-weights.md) for a verified, pinned model
download instead of copying a cache.

After reconnecting the camera, check RGB, depth, IR, and 3D and verify the reported
detector/material model selection. Physical camera operation must be checked
on the destination's USB controller and cable.
The installer does not migrate firmware or test destination hardware.

## Upgrade and remove

Build a higher version, install it with `sudo apt install ./NEW-PACKAGE.deb`,
close the owning console, run `bolt-rgb-setup`, then reopen the console.
Changed dependency pins select a new virtualenv; old environments remain
available under `~/.local/share/bolt-rgb-runtime/`.

```sh
sudo apt remove bolt-rgb
```

Removal deletes package-owned launchers, code, and USB rules. Both remove and
purge preserve all user archives, model caches, credentials, and virtualenvs.
An earlier checkout launcher in `~/.local/bin/rgb-console` or desktop entry in
`~/.local/share/applications/` may still take precedence over the packaged
entry. Use `/usr/bin/bolt-rgb` to select the installed package explicitly.


## ARM64 qualification on ventunoq

The November 6 ARM64 check used ventunoq running Ubuntu 24.04, Python 3.12,
and its native aarch64 wheels. Both the small package and the 508 MiB
bundled-wheel package installed through APT. Normal-user setup passed;
a fresh bundled setup also passed inside a network namespace with no network
interfaces, using only the package's verified wheels.

The installed Qt/WebEngine console rendered the real dashboard and Records
against the isolated synthetic service under Xvfb, then closed without stopping
the attached service. Camera SDK, ONNX Runtime, and OpenVINO imports passed.
No physical Bolt/D415 capture or depth accuracy was established on this host.
The test did not start a hardware camera service or perform cloud inference.
Private archives, credentials, and model caches were not packaged or migrated.
Native amd64 and ARM64 CI lanes repeat package building/offline setup, actual
model runtime checks, SDK initialization, and the X11 lifecycle check.
