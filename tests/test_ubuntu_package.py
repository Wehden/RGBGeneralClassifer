"""Installer contracts: actual .deb payload, offline setup, and user state."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("ubuntu_builder", ROOT / "scripts/build-ubuntu-package.py")
launcher = load("ubuntu_launcher", ROOT / "packaging/ubuntu/launcher.py")


@pytest.fixture
def application(tmp_path):
    root = tmp_path / "source"
    for name in ("rgb", "common", "board"):
        (root / name).mkdir(parents=True)
        (root / name / "__init__.py").write_text("")
    (root / "rgb/static").mkdir()
    (root / "rgb/static/app.js").write_text("// application\n")
    for name in builder.REQUIREMENTS:
        (root / name).write_text("# dependency pins\n")
    (root / "LICENSE").write_text("Application license\n")
    model = root / "models/yoloe-26n-pf-320"
    model.mkdir(parents=True)
    (model / "model.onnx").write_bytes(b"test-model")
    (model / "manifest.json").write_text(json.dumps({
        "sha256": hashlib.sha256(b"test-model").hexdigest()}))
    (model / "LICENSE").write_text("Model license\n")
    return root


def test_application_payload_excludes_private_state_and_keeps_model(application, tmp_path):
    (application / "runtime.env").write_text("SECRET=not-for-the-package")
    (application / ".venv").mkdir()
    (application / ".venv/credential.py").write_text("secret")
    (application / "rgb/archive.sqlite3").write_bytes(b"private capture archive")
    (application / "rgb/capture.jpg").write_bytes(b"private camera pixels")
    target = tmp_path / "payload"
    builder.stage_application(target, application)
    assert (target / "rgb/static/app.js").is_file()
    assert (target / "models/yoloe-26n-pf-320/model.onnx").read_bytes() == b"test-model"
    assert not any(path.suffix in {".env", ".sqlite3", ".jpg"} for path in target.rglob("*"))
    assert not (target / ".venv").exists()


def test_invalid_model_or_symlinked_source_cannot_enter_package(application, tmp_path):
    model = application / "models/yoloe-26n-pf-320/model.onnx"
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        builder.stage_application(tmp_path / "bad-model", application)
    model.write_bytes(b"test-model")
    (application / "rgb/credential.py").symlink_to(application / "LICENSE")
    with pytest.raises(ValueError, match="ordinary"):
        builder.stage_application(tmp_path / "bad-source", application)


@pytest.mark.skipif(not shutil.which("dpkg-deb"), reason="Debian package tools are unavailable")
def test_actual_deb_build_extracts_launchers_rules_and_checksum(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build-ubuntu-package.py"),
         "--version", "0.0.1", "--output-dir", str(tmp_path)], check=True, capture_output=True)
    package = tmp_path / "bolt-rgb_0.0.1_amd64.deb"
    fields = subprocess.check_output(
        ["dpkg-deb", "--field", str(package), "Package", "Architecture", "Depends"], text=True)
    assert "Package: bolt-rgb" in fields and "Architecture: amd64" in fields
    assert "python3.12-venv" in fields and "libxcb-cursor0" in fields
    subprocess.run(
        ["dpkg-deb", "--raw-extract", str(package), str(tmp_path / "extracted")], check=True)
    extracted = tmp_path / "extracted"
    assert (extracted / "usr/bin/bolt-rgb-service").stat().st_mode & 0o777 == 0o755
    assert "launcher.py service" in (extracted / "usr/bin/bolt-rgb-service").read_text()
    rule = (extracted / "usr/lib/udev/rules.d/71-bolt-rgb.rules").read_text()
    assert 'MODE="0660"' in rule and 'TAG+="uaccess"' in rule and '"066b"' in rule
    assert '"8086"' in rule and '"0ad3"' in rule
    assert not (extracted / "opt/bolt-rgb/.venv").exists()
    receipt = package.with_suffix(".deb.sha256").read_text().split()
    assert receipt == [hashlib.sha256(package.read_bytes()).hexdigest(), package.name]
    # Removal must never traverse or delete a user's capture/model directories.
    postrm = (extracted / "DEBIAN/postrm").read_text()
    assert "rm " not in postrm and "HOME" not in postrm


def test_offline_setup_requires_bundled_verified_wheels(application, tmp_path, monkeypatch):
    monkeypatch.setenv("BOLT_RGB_RUNTIME_DIR", str(tmp_path / "runtime"))
    with pytest.raises(ValueError, match="no bundled wheels"):
        launcher.setup(application, offline=True)
    (application / "wheels").mkdir()
    wheel = application / "wheels/example-1.0-py3-none-any.whl"
    wheel.write_bytes(b"valid-wheel")
    (application / "wheels.json").write_text(json.dumps({
        wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()}))
    wheel.write_bytes(b"tampered-wheel")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not invoke pip"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        launcher.setup(application, offline=True)
    wheel.write_bytes(b"valid-wheel")
    (application / "wheels/unlisted-1.0-py3-none-any.whl").write_bytes(b"unverified")
    with pytest.raises(ValueError, match="do not match"):
        launcher.setup(application, offline=True)


def test_failed_setup_is_not_ready_and_retry_keeps_archive(application, tmp_path, monkeypatch):
    monkeypatch.setenv("BOLT_RGB_RUNTIME_DIR", str(tmp_path / "runtime"))
    archive = tmp_path / "archive.sqlite3"
    archive.write_bytes(b"existing archive")
    calls = []

    def create(directory):
        (directory / "bin").mkdir(parents=True, exist_ok=True)
        (directory / "bin/python").touch()

    monkeypatch.setattr(launcher.venv.EnvBuilder, "create", lambda self, path: create(path))

    def fail_imports(command, **kwargs):
        calls.append(command)
        if "-c" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail_imports)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.setup(application)
    directory = launcher.runtime_directory(application)
    assert not launcher.is_ready(directory)
    assert archive.read_bytes() == b"existing archive"
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    assert launcher.setup(application) == directory and launcher.is_ready(directory)
    count = len(calls)
    assert launcher.setup(application) == directory and len(calls) == count
    # Changing dependency pins must select a fresh environment, never the old venv.
    (application / "requirements-desktop.txt").write_text("different==2.0\n")
    assert launcher.runtime_directory(application) != directory


def test_packaged_service_enforces_loopback_and_passes_user_options(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python").touch()
    (runtime / ".ready").touch()
    monkeypatch.setattr(launcher, "runtime_directory", lambda: runtime)
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(launcher.os, "umask", lambda value: None)
    monkeypatch.setattr(launcher.os, "chdir", lambda path: None)
    monkeypatch.setattr(launcher.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(launcher.sys, "argv",
                        ["launcher.py", "service", "--no-vlm", "--host", "0.0.0.0"])
    calls = []
    monkeypatch.setattr(launcher.os, "execv", lambda python, arguments: calls.append(arguments))
    launcher.main()
    assert calls[0][1:4] == ["-m", "rgb.local", "--no-vlm"]
    assert calls[0][-2:] == ["--host", "127.0.0.1"]
