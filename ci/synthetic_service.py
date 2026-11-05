"""Serve the real local console/API with synthetic frames and an isolated archive."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from waitress import create_server

from rgb import camera_devices
from rgb.camera import Camera, make_frame
from rgb.local_runtime import LocalRuntime
from rgb.local_store import LocalStore, encode
from rgb.store import now
from rgb.web import create_app


class SyntheticMaterials:
    """A UI fixture: the separate model lane qualifies real inference."""
    def classify(self, crop):
        return {'status': 'identified', 'label': 'plastic', 'best_candidate': 'plastic',
                'score': .94, 'similarity': .12, 'margin': .8,
                'score_type': 'supervised_material_head_score',
                'model': 'synthetic-ci-student', 'manifest_sha256': 'b' * 64,
                'inference_ms': 1., 'candidates': [{'label': 'plastic', 'score': .94}],
                'policy': {'minimum_probability': .8, 'minimum_margin': .05}}


def make_service(directory, depth=True, cameras=False):
    store = LocalStore(directory, analysis_enabled=False)
    camera = Camera(backend='orbbec' if depth else 'v4l2', freshness_s=600)
    camera.info = {'name': 'Synthetic RGB-D' if depth else 'Synthetic RGB'}
    calibration = {'schema_version': 1, 'planes': {'color': {
        'intrinsics': {'width': 320, 'height': 240, 'fx': 280., 'fy': 280., 'cx': 160., 'cy': 120.},
        'distortion': {'model': 'NONE'},
        'profile': {'width': 320, 'height': 240, 'fps': 15, 'format': 'synthetic'},
    }}}
    if not depth:
        calibration = {'schema_version': 1, 'available': False, 'backend': 'v4l2'}
    metadata = {
        'frame_id': str(uuid4()), 'session_id': str(uuid4()), 'source_id': 'femto-bolt',
        'sequence': 1, 'received_at': now(), 'received_monotonic_s': time.monotonic(),
        'timestamps_us': dict.fromkeys(['color', 'depth', 'ir'], 1000) if depth else {},
        'timestamp_domain': 'synthetic-ci', 'profiles': {'color': {'width': 320, 'height': 240}},
        'calibration': calibration, 'calibration_sha256': hashlib.sha256(encode(calibration).encode()).hexdigest(),
        'device': {'name': camera.name, 'serial': 'synthetic-ci'},
    }
    plane = np.full((240, 320), 1200, np.uint16)
    if depth:
        metadata.update(depth_scale_mm=1., registered_depth_scale_mm=1.)
        metadata['profiles'].update(depth={'width': 320, 'height': 240}, ir={'width': 320, 'height': 240})
    color = np.full((240, 320, 3), (40, 120, 60), np.uint8)
    frame = make_frame(color, plane if depth else None,
                       np.arange(240 * 320, dtype=np.uint16).reshape(240, 320) if depth else None,
                       plane if depth else None, metadata)
    camera.session_id = metadata['session_id']
    camera.calibration_hash = metadata['calibration_sha256']
    camera.publish(frame)
    detection = {'class_id': 44, 'label': 'bottle', 'confidence': .85,
                 'box': [.2, .3, .8, .7], 'track_id': str(uuid4())}
    if depth:
        # Synthetic UI reading; does not certify real volume estimation.
        detection['volume'] = {'schema_version': 1, 'status': 'conditional_estimate',
                               'estimate_m3': .00075, 'method': 'upright_cylinder',
                               'dimensions_m': [.07, .2, .07], 'native_samples': 100,
                               'source_frame_id': frame.id, 'warnings': ['synthetic CI fixture']}
    result = {'frame_id': frame.id, 'session_id': metadata['session_id'], 'observed_at': now(),
              'detector_ms': 1., 'detections': [detection],
              'detector': {'model': 'synthetic-ci-detector', 'sha256': 'a' * 64, 'config': {}}}
    camera.latest_detection = (frame, result)
    camera.detector_status = {'state': 'running', 'inference_ms': 1.}
    runtime = LocalRuntime(store, camera, local_materials=True)
    runtime._material_pass(SyntheticMaterials(), frame, result['detections'], result['detector'])
    runtime._attach_material(frame, detection)
    event = runtime.archive(frame, result)
    inventory = [{'id': backend, 'backend': backend,
                      'name': {'orbbec': 'Synthetic Bolt', 'realsense': 'Synthetic D415', 'v4l2': 'Synthetic webcam'}[backend],
                      'serial': backend, 'usb_serial': backend, 'device': '/dev/synthetic',
                      'connected': True} for backend in ['orbbec', 'realsense', 'v4l2']]
    if not cameras:
        inventory = [item for item in inventory if item['backend'] == camera.backend]
    camera_devices.discover = lambda: deepcopy(inventory)
    camera.selection_id = 'orbbec' if depth else 'v4l2'
    if cameras:
        def select_camera(selection):
            # Only sensor acquisition is synthetic; use the real selection API,
            # runtime reset, immutable planes, and browser controls.
            camera.stop_event.set()
            camera.stop_event = threading.Event()
            camera.backend = selection['backend']
            camera.selection_id = selection['id']
            camera.serial = camera.usb_serial = selection['usb_serial']
            camera.info = {'name': selection['name']}
            camera.session_id = str(uuid4())
            meta = {**metadata, 'frame_id': str(uuid4()), 'session_id': camera.session_id, 'device': camera.info}
            native_depth = plane.copy() if camera.backend != 'v4l2' else None
            native_ir = (plane.astype(np.uint8) if camera.backend == 'realsense'
                         else plane.copy() if native_depth is not None else None)
            fresh = make_frame(color.copy(), native_depth, native_ir,
                               plane.copy() if native_depth is not None else None, meta)
            with camera.condition:
                camera.frames.clear()
                camera.publish(fresh)
                selected_result = deepcopy(result)
                selected_result.update(frame_id=fresh.id, session_id=camera.session_id)
                if native_depth is None:
                    for detection in selected_result['detections']: detection.pop('volume', None)
                camera.latest_detection = (fresh, selected_result)
        camera.select = select_camera
    app = create_app(store, sources={'femto-bolt': {'local': True}}, local_runtime=runtime)
    return app, camera, store, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-only', action='store_true')
    parser.add_argument('--cameras', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='rgb-ci-console-') as directory:
        app, _, _, event = make_service(directory, depth=not args.rgb_only, cameras=args.cameras)
        server = create_server(app, host='127.0.0.1', port=0, threads=16)
        print(json.dumps({'url': f'http://127.0.0.1:{server.effective_port}',
                          'event_id': event['event_id']}), flush=True)
        server.run()


if __name__ == '__main__':
    main()
