"""Per-user runtime setup and launch for the local Ubuntu package."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv


PACKAGE_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ("requirements.txt", "requirements-local.txt", "requirements-desktop.txt")


def native_architecture():
    aliases = {"x86_64":"amd64", "amd64":"amd64", "aarch64":"arm64", "arm64":"arm64"}
    try:
        return aliases[platform.machine().lower()]
    except KeyError:
        raise ValueError("Supported runtime architectures are amd64 and arm64") from None


def validate_target(root):
    target = json.loads((root / "package-target.json").read_text())
    release = platform.freedesktop_os_release()
    if target != {"schema_version":1, "architecture":native_architecture(), "python":"3.12",
                  "distribution":"ubuntu", "release":"24.04"}:
        raise ValueError("Package target does not match this host; install its native amd64/arm64 package")
    if sys.version_info[:2] != (3, 12) or release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        raise ValueError("This package requires Python 3.12 on Ubuntu 24.04 amd64 or arm64")


def runtime_directory(root=PACKAGE_ROOT):
    """Do not share or relocate the checkout's virtualenv."""
    digest = hashlib.sha256()
    digest.update(b"native-qt-validation-v2")
    digest.update(sys.version.encode())
    digest.update(native_architecture().encode())
    digest.update((root / "package-target.json").read_bytes())
    for name in REQUIREMENTS:
        digest.update((root / name).read_bytes())
    wheels = root / "wheels.json"
    if wheels.exists():
        digest.update(wheels.read_bytes())
    base = Path(os.environ.get(
        "BOLT_RGB_RUNTIME_DIR", str(Path.home() / ".local/share/bolt-rgb-runtime"))).expanduser()
    return base / ("python-3.12-" + native_architecture() + "-" + digest.hexdigest()[:20])


def is_ready(directory):
    return (directory / ".ready").is_file() and (directory / "bin/python").is_file()


def verify_wheels(root):
    """Reject incomplete or changed bundled wheels before invoking pip."""
    manifest = json.loads((root / "wheels.json").read_text())
    if not manifest:
        raise ValueError("The bundled wheel manifest is empty")
    if {path.name for path in (root / "wheels").glob("*.whl")} != set(manifest):
        raise ValueError("The bundled wheels do not match the wheel manifest")
    for name, expected in manifest.items():
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("Invalid bundled wheel filename")
        path = root / "wheels" / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Bundled wheel is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Bundled wheel checksum mismatch: {name}")
    return manifest


def setup(root=PACKAGE_ROOT, *, offline=False):
    validate_target(root)
    bundled = (root / "wheels.json").exists()
    if offline and not bundled:
        raise ValueError("This package has no bundled wheels; use a package built with --bundle-wheels")
    directory = runtime_directory(root)
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Two setup invocations must not install into the same environment together.
    with (directory.parent / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if is_ready(directory):
            print(f"Runtime is ready: {directory}", flush=True)
            return directory
        if bundled:
            verify_wheels(root)
        print(f"Preparing the user runtime in {directory}", flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
        python = str(directory / "bin/python")
        command = [python, "-m", "pip", "--isolated", "--disable-pip-version-check",
                   "install", "--only-binary=:all:"]
        if bundled:
            command += ["--no-index", "--find-links", str(root / "wheels")]
        command += ["-r", str(root / "requirements-desktop.txt")]
        subprocess.run(command, check=True)
        subprocess.run([python, "-m", "pip", "check"], check=True)
        # Import the actual local runtime; never acquire the camera or invoke a VLM.
        subprocess.run(
            [python, "-c", "import rgb.local, PySide6, pyorbbecsdk, pyrealsense2, onnxruntime, openvino; from PySide6.QtWebEngineCore import QWebEnginePage; from PySide6.QtWebEngineWidgets import QWebEngineView"],
            cwd=root, check=True)
        (directory / ".ready").write_text("Dependencies installed and imports verified.\n")
    print("Setup complete. Launch bolt-rgb, or bolt-rgb-service --no-vlm.", flush=True)
    return directory


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"setup", "desktop", "service"}:
        raise SystemExit("Use bolt-rgb-setup, bolt-rgb, or bolt-rgb-service.")
    mode, arguments = sys.argv[1], sys.argv[2:]
    offline = False
    if mode == "setup":
        parser = argparse.ArgumentParser(description="Prepare RGB Console's per-user Python environment")
        parser.add_argument("--offline", action="store_true",
                            help="Require bundled wheels; never contact a Python package index")
        offline = parser.parse_args(arguments).offline
    if os.geteuid() == 0:
        raise SystemExit("Run this command as your normal user, without sudo.")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("This package requires Python 3.12 on Ubuntu 24.04 amd64 or arm64.")
    # Restrict newly created user state, logs, and captures.
    os.umask(0o077)
    try:
        validate_target(PACKAGE_ROOT)
        if mode == "setup":
            setup(offline=offline)
            return
        directory = runtime_directory()
        if not is_ready(directory):
            message = "Run bolt-rgb-setup in a terminal before opening RGB Console."
            if mode == "desktop" and os.environ.get("DISPLAY"):
                # Tk is optional: the terminal command always reports the same error.
                try:
                    import tkinter
                    from tkinter import messagebox
                    window = tkinter.Tk()
                    window.withdraw()
                    messagebox.showerror("RGB Console setup", message)
                    window.destroy()
                except ImportError:
                    pass
                except tkinter.TclError:
                    pass
            raise SystemExit(message)
        os.chdir(PACKAGE_ROOT)
        module = "rgb.desktop" if mode == "desktop" else "rgb.local"
        if mode == "service":
            # Last CLI value wins over the environment and earlier --host options.
            arguments += ["--host", "127.0.0.1"]
        os.execv(str(directory / "bin/python"),
                 [str(directory / "bin/python"), "-m", module, *arguments])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"RGB Console: {exc}") from exc


if __name__ == "__main__":
    main()
