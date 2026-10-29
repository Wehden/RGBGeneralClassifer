"""Experimental YOLOE mask decoding; no change to the live detector contract."""
import math

import cv2
import numpy as np

from board.yoloe import decode


def instances(output, prototypes, labels, shape, transform, threshold=.45):
    """Decode original-image masks with Ultralytics' native scaling convention."""
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 38:
        raise ValueError('Mask coefficients require [1,N,38] detections')
    if prototypes.ndim != 4 or prototypes.shape[:2] != (1, 32):
        raise ValueError('Expected [1,32,H,W] mask prototypes')
    if not np.isfinite(prototypes).all():
        raise ValueError('Nonfinite mask prototypes')
    h, w = shape[:2]
    _, channels, mh, mw = prototypes.shape
    gain = min(4 * mh / h, 4 * mw / w)
    pad = (4 * mh - max(round(h * gain), 1)) | (4 * mw - max(round(w * gain), 1))
    factor = 4 // math.gcd(4, pad | pad // 2)
    found = []
    for row_index, row in enumerate(output[0]):
        readings = decode(row[None, None], labels, shape, transform, threshold)
        if not readings or not np.isfinite(row[6:]).all():
            continue
        logits = (row[6:] @ prototypes[0].reshape(channels, -1)).reshape(mh, mw)
        if factor > 1:
            logits = cv2.resize(logits, (factor * mw, factor * mh), interpolation=cv2.INTER_LINEAR)
        sh, sw = logits.shape
        mask_gain = min(sh / h, sw / w)
        left = round((sw - max(round(w * mask_gain), 1)) / 2 - .1)
        top = round((sh - max(round(h * mask_gain), 1)) / 2 - .1)
        content = logits[top:top + max(round(h * mask_gain), 1),
                         left:left + max(round(w * mask_gain), 1)]
        mask = cv2.resize(content, (w, h), interpolation=cv2.INTER_LINEAR) > 0
        reading = readings[0]
        y1, x1, y2, x2 = reading['box']
        mask &= ((np.arange(w)[None] >= x1 * w) & (np.arange(w)[None] < x2 * w))
        mask &= ((np.arange(h)[:, None] >= y1 * h) & (np.arange(h)[:, None] < y2 * h))
        found.append({**reading, 'row_index': row_index, 'mask': mask})
    return found
