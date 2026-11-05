"""Passive discovery of physical cameras, without opening capture streams."""
import hashlib
from pathlib import Path


USB_ROOT = Path('/sys/bus/usb/devices')
VIDEO_ROOT = Path('/sys/class/video4linux')


def identifier(backend, physical):
    return backend + ':' + hashlib.sha256(physical.encode()).hexdigest()[:24]


def usb_identity(path):
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / 'idVendor').is_file() and (parent / 'idProduct').is_file():
            serial = (parent / 'serial').read_text().strip() if (parent / 'serial').exists() else ''
            return serial or 'usb:' + parent.name
    raise OSError('No physical USB camera identity')


def discover():
    cameras = []
    native = set()
    for path in sorted(USB_ROOT.glob('*')):
        try:
            vendor = (path / 'idVendor').read_text().strip().lower()
            product = (path / 'idProduct').read_text().strip().lower()
            name = (path / 'product').read_text().strip()
            backend = ('orbbec' if (vendor, product) == ('2bc5', '066b')
                       else 'realsense' if vendor == '8086' and 'realsense' in name.lower() else None)
            if backend is None:
                continue
            physical = usb_identity(path)
            native.add(physical)
            if physical.startswith('usb:'):
                continue  # Native SDK selection needs a verified serial/ASIC identity.
            label = 'Femto Bolt' if backend == 'orbbec' else 'RealSense D415' if '415' in name else name
            cameras.append({'id': identifier(backend, physical), 'name': label, 'backend': backend,
                            'serial': physical if backend == 'orbbec' else None,
                            'usb_serial': physical, 'device': None, 'connected': True})
        except OSError:
            continue
    seen = set(native)
    for node in sorted(VIDEO_ROOT.glob('video*')):
        try:
            if (node / 'index').read_text().strip() != '0':
                continue
            physical = usb_identity(node / 'device')
            if physical in seen:
                continue
            name = (node / 'name').read_text().strip() or 'USB camera'
            seen.add(physical)
            cameras.append({'id': identifier('v4l2', physical), 'name': name, 'backend': 'v4l2',
                            'serial': physical, 'usb_serial': physical,
                            'device': '/dev/' + node.name, 'connected': True})
        except OSError:
            continue
    return cameras


def catalog(camera):
    cameras = discover()
    physical = getattr(camera, 'usb_serial', None) or camera.info.get('usb_serial') or camera.serial
    active = camera.selection_id
    if active is None:
        active = next((item['id'] for item in cameras if item['backend'] == camera.backend
                       and (item['usb_serial'] == physical
                            or camera.backend == 'v4l2' and item['device'] == camera.device)), None)
    if active is None:
        active = identifier(camera.backend, physical or camera.device)
    if not any(item['id'] == active for item in cameras):
        cameras.append({'id': active, 'name': camera.name, 'backend': camera.backend,
                        'serial': camera.serial, 'usb_serial': physical,
                        'device': camera.device if camera.backend == 'v4l2' else None,
                        'connected': False})
    return {'cameras': cameras, 'active_id': active, 'state': camera.state, 'error': camera.error}


def choose_default(camera):
    """Start one camera of the requested backend so multi-camera UI remains reachable."""
    if camera.serial is not None or camera.backend == 'v4l2':
        return
    candidates = [item for item in discover() if item['backend'] == camera.backend]
    if candidates:
        selection = candidates[0]
        camera.selection_id = selection['id']
        camera.serial = selection['serial']
        camera.usb_serial = selection['usb_serial']
        camera.info = {'name': selection['name']}
