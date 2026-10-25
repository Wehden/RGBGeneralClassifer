"""One-to-one box association for continuous camera detections."""
from uuid import uuid4
from common.geometry import iou


class Tracker:
    def __init__(self, max_gap_s=3):
        self.tracks = {}
        self.max_gap_s = max_gap_s

    def update(self, detections, timestamp):
        self.tracks = {key: track for key, track in self.tracks.items() if timestamp-track['last_seen'] <= self.max_gap_s}
        pairs = sorted(((iou(d['box'], t['box']), index, key)
                        for index, d in enumerate(detections) for key, t in self.tracks.items()
                        if d['class_id'] == t['class_id']), reverse=True)
        assigned, used = set(), set()
        for overlap, index, key in pairs:
            if overlap < .5 or index in assigned or key in used:
                continue
            detections[index]['track_id'] = key
            assigned.add(index)
            used.add(key)
        for d in detections:
            d.setdefault('track_id', str(uuid4()))
            self.tracks[d['track_id']] = {'class_id': d['class_id'], 'box': d['box'], 'last_seen': timestamp}
        return detections
