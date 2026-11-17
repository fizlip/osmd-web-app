"""Bright-spot ("blob") detection for camera frames.

A blob is a connected region of saturated, near-white pixels, e.g. the core
of a laser spot. Coloured speckle around it is ignored because only pixels
whose grey level reaches the threshold count, and specks smaller than
min_area are dropped. Frames are numpy BGR arrays (OpenCV's layout, and what
Picamera2 returns for the "RGB888" format).
"""

from dataclasses import dataclass

import cv2
import numpy as np

THRESHOLD = 240   # grey level (0-255) a pixel must reach to be "saturated"
MIN_AREA = 100    # pixels; smaller bright regions are treated as glints


@dataclass
class Blob:
    x: float        # centroid, pixels from the left edge
    y: float        # centroid, pixels from the top edge
    area: int       # pixels
    bbox: tuple     # (left, top, width, height)

    @property
    def radius(self) -> float:
        """Radius of a circle with the same area."""
        return float(np.sqrt(self.area / np.pi))


def detect_blobs(frame, threshold=THRESHOLD, min_area=MIN_AREA):
    """Return the bright blobs in a BGR frame, largest first."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, threshold - 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill pinholes
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = [Blob(float(cx), float(cy), int(s[cv2.CC_STAT_AREA]),
                  tuple(int(v) for v in s[:4]))
             for s, (cx, cy) in zip(stats[1:n], centroids[1:n])
             if s[cv2.CC_STAT_AREA] >= min_area]
    return sorted(blobs, key=lambda b: b.area, reverse=True)


def annotate(frame, blobs):
    """Draw each blob's outline, centroid and (x, y) label onto frame."""
    for b in blobs:
        c = (round(b.x), round(b.y))
        cv2.circle(frame, c, max(round(b.radius), 4), (0, 0, 255), 2)
        cv2.drawMarker(frame, c, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        label = f"({b.x:.0f}, {b.y:.0f})"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3)
        x = c[0] + round(b.radius) + 6
        if x + tw > frame.shape[1]:  # no room on the right: label on the left
            x = c[0] - round(b.radius) - 6 - tw
        org = (x, c[1] + 5)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process(frame, threshold=THRESHOLD, min_area=MIN_AREA):
    """Detect blobs, draw them, and JPEG-encode. -> (jpeg_bytes, blobs)"""
    blobs = detect_blobs(frame, threshold, min_area)
    ok, jpeg = cv2.imencode(".jpg", annotate(frame, blobs))
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpeg.tobytes(), blobs


def synthetic_frame(w, h, t):
    """Test frame for --simulate: a white spot circling over green speckle."""
    rng = np.random.default_rng(int(t * 10))
    frame = np.zeros((h, w, 3), np.uint8)
    frame[..., 1] = rng.integers(0, 200, (h, w), dtype=np.uint8)
    cx = int(w / 2 + w / 4 * np.cos(t * 0.5))
    cy = int(h / 2 + h / 4 * np.sin(t * 0.5))
    cv2.circle(frame, (cx, cy), min(w, h) // 12, (255, 255, 255), -1)
    return frame
