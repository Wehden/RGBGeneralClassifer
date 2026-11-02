"""Offline material crop ablations; the live console preprocessing is unchanged."""
import cv2
import numpy as np


def material_crop(rgb, bounds, strategy, mask=None, size=384, minimum_patch=24):
    """Return exact student pixels and the source bounds used for their evidence.

    ``bounds`` are integer XYXY source coordinates. Masks describe the same
    source frame. An interior patch uses the largest mask-interior square,
    rather than assuming the centre of an object box is foreground.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB source')
    if strategy not in {'stretch', 'letterbox', 'foreground', 'interior'}:
        raise ValueError('Unknown crop strategy')
    h, w = rgb.shape[:2]
    if len(bounds) != 4 or any(type(v) is not int for v in bounds):
        raise ValueError('Expected integer XYXY bounds')
    x1, y1, x2, y2 = bounds
    if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
        raise ValueError('Crop bounds outside source')
    if size < 1 or minimum_patch < 1:
        raise ValueError('Invalid crop size')
    patch = rgb[y1:y2, x1:x2].copy()
    source_bounds = list(bounds)
    if strategy in {'foreground', 'interior'}:
        if mask is None or mask.shape != (h, w) or mask.dtype != np.bool_:
            raise ValueError('Expected source-aligned boolean mask')
        foreground = mask[y1:y2, x1:x2]
        if not foreground.any():
            raise ValueError('No foreground evidence')
        if strategy == 'foreground':
            patch[~foreground] = 127
        else:
            # Pad with background: a full foreground box must still have a
            # finite distance to its boundary. Chebyshev distance fits squares.
            padded = np.pad(foreground.astype(np.uint8), 1)
            distance = cv2.distanceTransform(padded, cv2.DIST_C, 3)[1:-1, 1:-1]
            cy, cx = np.unravel_index(distance.argmax(), distance.shape)
            radius = int(distance[cy, cx]) - 1
            side = 2 * radius + 1
            if side < minimum_patch:
                raise ValueError('Insufficient contiguous surface evidence')
            left, top = int(cx) - radius, int(cy) - radius
            patch = patch[top:top+side, left:left+side]
            if not foreground[top:top+side, left:left+side].all():
                raise ValueError('Interior crop contains background')
            source_bounds = [x1+left, y1+top, x1+left+side, y1+top+side]
    if strategy in {'stretch', 'interior'}:
        pixels = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
    else:
        ph, pw = patch.shape[:2]
        gain = min(size/pw, size/ph)
        rw, rh = max(1, round(pw*gain)), max(1, round(ph*gain))
        pixels = np.full((size, size, 3), 127, np.uint8)
        left, top = (size-rw)//2, (size-rh)//2
        pixels[top:top+rh, left:left+rw] = cv2.resize(
            patch, (rw, rh), interpolation=cv2.INTER_CUBIC)
    return pixels, source_bounds
