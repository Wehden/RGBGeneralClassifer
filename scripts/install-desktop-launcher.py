#!/usr/bin/env python3
"""Install a desktop menu entry and user PATH command for this checkout."""
import os
import sys
from pathlib import Path


def quoted(value):
    # Desktop Exec quoting is parsed by the desktop, without a shell.
    if '\n' in value or '\r' in value:
        raise ValueError('Desktop launcher paths cannot contain newlines')
    return '"' + ''.join('\\' + char if char in '\\`"$' else char for char in value).replace('%', '%%') + '"'


def main():
    root = Path(__file__).resolve().parent.parent
    python = root / '.venv/bin/python'
    if not python.exists():
        raise SystemExit('Install requirements-desktop.txt into .venv first.')
    commands = Path.home() / '.local/bin'
    commands.mkdir(parents=True, exist_ok=True)
    command = commands / 'rgb-console'
    helper = root / 'scripts/rgb-console'
    if os.path.lexists(command) and (not command.is_symlink() or command.resolve() != helper):
        raise SystemExit(f'{command} already exists and belongs to another launcher.')
    helper.chmod(helper.stat().st_mode | 0o111)
    if not command.is_symlink():
        command.symlink_to(helper)
    directory = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'applications'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'rgb-console.desktop'
    target.write_text(
        '[Desktop Entry]\nType=Application\nName=RGB Console\n'
        'Comment=Camera capture and visual review\n'
        f'Exec={quoted(str(python))} -m rgb.desktop\n'
        f'Path={root}\nIcon=camera-video\nTerminal=false\nCategories=Graphics;Photography;\n')
    legacy = directory / 'bolt-rgb.desktop'
    if legacy.exists() and f'Exec={quoted(str(python))} -m rgb.desktop\n' in legacy.read_text():
        legacy.unlink()
    print(target)
    print(command)


if __name__ == '__main__':
    main()
