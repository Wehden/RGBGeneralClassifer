#!/usr/bin/env python3
"""Build a local Ubuntu 24.04 amd64 .deb without sudo or private runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "packaging/ubuntu"
REQUIREMENTS = ("requirements.txt", "requirements-local.txt", "requirements-desktop.txt")


def copy_file(source, destination, *, executable=False):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected an ordinary package source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o755 if executable else 0o644)


def stage_application(destination, root=ROOT):
    # Allowlist only application code, UI assets, requirements, and the pinned
    # fallback detector. No recursive copy of the checkout or private model cache.
    suffixes = {".py", ".html", ".css", ".js"}
    for name in ("rgb", "common", "board"):
        for source in sorted((root / name).rglob("*")):
            relative = source.relative_to(root)
            if "__pycache__" not in relative.parts and source.suffix in suffixes:
                copy_file(source, destination / relative)
    for name in (*REQUIREMENTS, "LICENSE"):
        copy_file(root / name, destination / name)
    model = Path("models/yoloe-26n-pf-320")
    for name in ("model.onnx", "manifest.json", "LICENSE"):
        copy_file(root / model / name, destination / model / name)
    manifest = json.loads((destination / model / "manifest.json").read_text())
    with (destination / model / "model.onnx").open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != manifest["sha256"]:
        raise ValueError("The fallback detector does not match its manifest checksum")


def bundle_wheels(application):
    """Resolve native wheels on the supported build host, never copy a venv."""
    if sys.version_info[:2] != (3, 12) or platform.machine() != "x86_64":
        raise ValueError("--bundle-wheels requires Python 3.12 on an amd64 Linux build host")
    release = platform.freedesktop_os_release()
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        raise ValueError("Build bundled wheels on Ubuntu 24.04 to match the destination")
    wheels = application / "wheels"
    subprocess.run(
        [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
         "download", "--only-binary=:all:", "--dest", str(wheels),
         "-r", str(application / "requirements-desktop.txt")], check=True)
    manifest = {}
    for path in sorted(wheels.glob("*.whl")):
        with path.open("rb") as stream:
            manifest[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not manifest:
        raise ValueError("No wheels were downloaded")
    (application / "wheels.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def stage_package(stage, version, *, bundled=False, root=ROOT):
    application = stage / "opt/bolt-rgb"
    stage_application(application, root)
    copy_file(TEMPLATES / "launcher.py", application / "launcher.py")
    if bundled:
        bundle_wheels(application)
    for command, mode in (("bolt-rgb", "desktop"), ("rgb-console", "desktop"),
                          ("bolt-rgb-service", "service"), ("bolt-rgb-setup", "setup")):
        path = stage / "usr/bin" / command
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '#!/bin/sh\nexec /usr/bin/python3.12 /opt/bolt-rgb/launcher.py '
            + mode + ' "$@"\n')
        path.chmod(0o755)
    copy_file(TEMPLATES / "bolt-rgb.desktop",
              stage / "usr/share/applications/bolt-rgb.desktop")
    copy_file(TEMPLATES / "71-bolt-rgb.rules",
              stage / "usr/lib/udev/rules.d/71-bolt-rgb.rules")
    copy_file(root / "LICENSE", stage / "usr/share/doc/bolt-rgb/copyright")
    copy_file(root / "docs/ubuntu-installer.md", stage / "usr/share/doc/bolt-rgb/README.md")
    # The model has a separate license from the application.
    copy_file(root / "models/yoloe-26n-pf-320/LICENSE",
              stage / "usr/share/doc/bolt-rgb/model-license")
    control = stage / "DEBIAN"
    control.mkdir()
    for name in ("postinst", "postrm"):
        copy_file(TEMPLATES / name, control / name, executable=True)
    payload = [path for path in stage.rglob("*") if path.is_file() and control not in path.parents]
    size = sum(path.stat().st_size for path in payload)
    (control / "control").write_text((TEMPLATES / "control").read_text().format(
        version=version, installed_size=(size + 1023) // 1024))
    checksums = []
    for path in sorted(payload):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        checksums.append(f"{digest}  {path.relative_to(stage)}\n")
    (control / "md5sums").write_text("".join(checksums))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=datetime.now(timezone.utc).strftime("0.%Y%m%d.1"),
                        help="Debian package version (default: 0.YYYYMMDD.1)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--bundle-wheels", action="store_true",
                        help="Include verified native Python wheels for offline user setup")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+~]*", args.version):
        parser.error("--version must start with a digit and contain only letters, digits, . + or ~")
    if not shutil.which("dpkg-deb"):
        parser.error("dpkg-deb is required (Ubuntu package: dpkg)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir.resolve() / f"bolt-rgb_{args.version}_amd64.deb"
    # A failed build must leave an existing installer intact.
    with tempfile.TemporaryDirectory(prefix="bolt-rgb-deb-") as temporary:
        stage = Path(temporary) / "package"
        stage_package(stage, args.version, bundled=args.bundle_wheels)
        pending = Path(temporary) / target.name
        subprocess.run(
            ["dpkg-deb", "--root-owner-group", "-Zzstd", "-z3", "--build",
             str(stage), str(pending)], check=True)
        shutil.copyfile(pending, target.with_suffix(".deb.tmp"))
        target.with_suffix(".deb.tmp").replace(target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    target.with_suffix(".deb.sha256").write_text(f"{digest}  {target.name}\n")
    print(f"Built {target} ({target.stat().st_size / 1024 / 1024:.1f} MiB)")
    print("Copy the .deb and its .sha256 receipt to the destination.")


if __name__ == "__main__":
    main()
