"""Use deployment-image interpreters against every shipped board model offline."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from board.detector import Detector as LiteDetector, overlay
from board.yoloe import Detector as YoloDetector

assert os.getuid() == 1000
ROOT = Path('/app/models')
assert hashlib.sha256((ROOT / 'efficientdet-lite0.tflite').read_bytes()).hexdigest() == (
    '2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b')
detectors = [LiteDetector(ROOT / 'efficientdet-lite0.tflite', ROOT / 'labelmap.txt', threshold=0)]
for name in ['yoloe-26n-pf-320', 'yoloe-26n-materials-320']:
    detectors.append(YoloDetector(ROOT / name / 'model.onnx', ROOT / name / 'manifest.json', threshold=0))
image = np.random.default_rng(7).integers(0, 256, (180, 320, 3), dtype=np.uint8)
for detector in detectors:
    detections, elapsed = detector.detect(image)
    assert detections and elapsed >= 0
    for item in detections:
        y1, x1, y2, x2 = item['box']
        assert 0 <= y1 < y2 <= 1 and 0 <= x1 < x2 <= 1
        assert 0 <= item['confidence'] <= 1
    json.dumps(detections, allow_nan=False)
    assert overlay(image, detections).shape == image.shape
print('PASS native nonroot board image: EfficientDet LiteRT and both pinned YOLOE ONNX models')
