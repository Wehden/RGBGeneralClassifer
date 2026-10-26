#!/usr/bin/env python3
"""Install Ubuntu's missing XCB cursor library into the active virtualenv."""
import ctypes
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Run this helper with the project virtualenv Python.')
    try:
        ctypes.CDLL('libxcb-cursor.so.0')
        print('The system XCB cursor library is already installed.')
        return
    except OSError:
        pass
    metadata = subprocess.check_output(['apt-cache', 'show', 'libxcb-cursor0'], text=True)
    entries = metadata.split('\n\n')
    ubuntu = None
    for entry in entries:
        fields = dict(line.split(': ', 1) for line in entry.splitlines()
                      if ': ' in line and not line.startswith(' '))
        if fields.get('Origin') == 'Ubuntu' and fields.get('SHA256') and fields.get('Filename'):
            ubuntu = fields
            break
    if ubuntu is None:
        raise SystemExit('This helper requires Ubuntu package metadata. '
                         'Install libxcb-cursor0 with your distribution package manager.')
    filename = ubuntu['Filename']
    if not filename.startswith('pool/') or '..' in filename:
        raise SystemExit('Unexpected Ubuntu package path')
    with urlopen('https://archive.ubuntu.com/ubuntu/' + filename, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != ubuntu['SHA256']:
        raise SystemExit('Ubuntu package checksum mismatch; no library was installed')
    with tempfile.TemporaryDirectory(prefix='bolt-desktop-libs-') as temporary:
        directory = Path(temporary)
        package = directory / 'cursor.deb'
        package.write_bytes(data)
        subprocess.run(['dpkg-deb', '-x', str(package), str(directory / 'unpacked')], check=True)
        libraries = list((directory / 'unpacked').rglob('libxcb-cursor.so*'))
        if not libraries:
            raise SystemExit('The package contains no XCB cursor library')
        destination = Path(sys.prefix) / 'lib'
        for library in libraries:
            if library.is_file():
                shutil.copy2(library, destination / library.name, follow_symlinks=True)
        print(f"Installed Ubuntu {ubuntu['Version']} in {destination}; no system files changed.")
    # Other XCB dependencies are ordinary desktop packages, not included here.
    ctypes.CDLL(str(destination / 'libxcb-cursor.so.0'), mode=ctypes.RTLD_GLOBAL)


if __name__ == '__main__':
    main()
