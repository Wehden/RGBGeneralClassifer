"""CPU LiteRT detector. Box coordinates are normalized y1,x1,y2,x2."""
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


class Detector:
    def __init__(self, model, labels, threshold=0.45):
        self.labels = Path(labels).read_text().splitlines()
        self.threshold = threshold
        self.interpreter = Interpreter(model_path=model, num_threads=int(os.getenv("INFERENCE_THREADS", "2")))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = self.interpreter.get_output_details()
        # This adapter deliberately accepts only the pinned EfficientDet contract.
        if self.input['shape'].tolist() != [1, 320, 320, 3] or self.input['dtype'] != np.uint8:
            raise ValueError('Expected EfficientDet-Lite0 uint8 [1,320,320,3] input')
        shapes = [x['shape'].tolist() for x in self.outputs]
        if shapes != [[1, 25, 4], [1, 25], [1, 25], [1]]:
            raise ValueError(f'Unexpected detector output contract: {shapes}')

    def detect(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (320, 320)), cv2.COLOR_BGR2RGB)
        self.interpreter.set_tensor(self.input['index'], rgb[np.newaxis])
        start = time.monotonic()
        self.interpreter.invoke()
        ms = (time.monotonic() - start) * 1000
        boxes, classes, scores, count = [self.interpreter.get_tensor(o['index'])[0] for o in self.outputs]
        found = []
        for i in range(min(int(count), len(scores))):
            score = float(scores[i])
            if score < self.threshold:
                continue
            cls = int(classes[i])
            if not 0 <= cls < len(self.labels):
                continue
            box = np.clip(boxes[i], 0, 1).tolist()
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            found.append({'class_id': cls, 'label': self.labels[cls], 'confidence': round(score, 4), 'box': box})
        return found, round(ms, 2)


def overlay(frame, detections):
    result = frame.copy()
    h, w = result.shape[:2]
    for d in detections:
        y1, x1, y2, x2 = d['box']
        p1, p2 = (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h))
        cv2.rectangle(result, p1, p2, (70, 230, 160), 2)
        cv2.putText(result, f"{d.get('display_label', d['label'])} {d.get('display_confidence', d['confidence']):.0%}", (p1[0], max(20, p1[1]-7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 230, 160), 2)
    return result
