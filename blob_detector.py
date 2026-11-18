"""Bright-spot ("blob") detection on the camera's raw 10-bit sensor data.

A blob is a connected region of saturated pixels, e.g. the white core of a
laser spot. Detection runs on the raw Bayer frame (one colour sample per
pixel, 0-1023 for a 10-bit sensor), before the ISP's tone curve and 8-bit
conversion, so "saturated" means the sensor itself clipped. A 2x2 Bayer cell
(R, G, G, B) counts only when all four samples reach the threshold, so
green-only speckle is ignored even where the green sites clip. Blob
coordinates are mapped into the processed preview frame (BGR, what
Picamera2 returns for "RGB888") so they can be drawn on the stream.
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

THRESHOLD = 700  # raw level (0-1023 at 10 bits) all 4 samples of a cell must reach
MIN_AREA = 80    # preview-frame pixels; smaller bright regions are glints


@dataclass
class Blob:
    x: float        # centroid, preview-frame pixels from the left edge
    y: float        # centroid, preview-frame pixels from the top edge
    area: int       # preview-frame pixels
    bbox: tuple     # (left, top, width, height), preview-frame pixels

    @property
    def radius(self) -> float:
        """Radius of a circle with the same area."""
        return float(np.sqrt(self.area / np.pi))


def raw_to_frame(raw_size, crop_limits, scaler_crop, frame_size):
    """Affine map from raw pixel coordinates to preview frame coordinates.

    crop_limits is the sensor area the raw mode reads out and scaler_crop the
    sensor area the ISP scaled into the frame, both (x, y, w, h) in
    full-sensor pixels. -> ((kx, ky), (ox, oy)) with frame = raw * k + o
    """
    k, o = [], []
    for i in (0, 1):
        to_sensor = crop_limits[2 + i] / raw_size[i]
        to_frame = frame_size[i] / scaler_crop[2 + i]
        k.append(to_sensor * to_frame)
        # pixel centres sit half a pixel in from the edge in both grids
        o.append((crop_limits[i] + 0.5 * to_sensor - scaler_crop[i])
                 * to_frame - 0.5)
    return tuple(k), tuple(o)


def detect_blobs(bayer, threshold=THRESHOLD, min_area=MIN_AREA,
                 scale=(1.0, 1.0), offset=(0.0, 0.0)):
    """Return the saturated blobs in a raw Bayer frame (uint16 sensor
    levels), largest first, mapped to the frame by raw * scale + offset."""
    h, w = bayer.shape[0] // 2 * 2, bayer.shape[1] // 2 * 2
    cells = bayer[:h, :w].reshape(h // 2, 2, w // 2, 2).min(axis=(1, 3))
    mask = np.where(cells >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill pinholes
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # cell c spans raw pixels 2c and 2c+1, so its centre is raw 2c + 0.5
    kx, ky = 2 * scale[0], 2 * scale[1]
    ox, oy = offset[0] + 0.5 * scale[0], offset[1] + 0.5 * scale[1]
    blobs = []
    for (left, top, bw, bh, cells_area), (cx, cy) in zip(stats[1:n],
                                                          centroids[1:n]):
        area = cells_area * kx * ky
        if area >= min_area:
            blobs.append(Blob(float(cx * kx + ox), float(cy * ky + oy),
                              round(area),
                              (round(left * kx + ox), round(top * ky + oy),
                               round(bw * kx), round(bh * ky))))
    return sorted(blobs, key=lambda b: b.area, reverse=True)


def annotate(frame, blobs):
    """Draw each blob's outline, centroid and (x, y) label onto frame.
    Labels are relative to the frame centre, so (0, 0) is mid-viewport."""
    h, w = frame.shape[:2]
    for b in blobs:
        c = (round(b.x), round(b.y))
        cv2.circle(frame, c, max(round(b.radius), 4), (0, 0, 255), 2)
        cv2.drawMarker(frame, c, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        label = f"({b.x - w / 2:.0f}, {b.y - h / 2:.0f})"
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


BAYER_OFFSETS = {
    # (row, col) sites of each colour's samples inside the 2x2 cell
    "RGGB": {"r": [(0, 0)], "g": [(0, 1), (1, 0)], "b": [(1, 1)]},
    "BGGR": {"b": [(0, 0)], "g": [(0, 1), (1, 0)], "r": [(1, 1)]},
    "GRBG": {"g": [(0, 0), (1, 1)], "r": [(0, 1)], "b": [(1, 0)]},
    "GBRG": {"g": [(0, 0), (1, 1)], "b": [(0, 1)], "r": [(1, 0)]},
}


def bayer_pattern(format_name):
    """Bayer colour order ("RGGB" etc.) from a raw format like "SRGGB16"."""
    m = re.search(r"RGGB|BGGR|GRBG|GBRG", format_name)
    if not m:
        raise ValueError(f"cannot determine Bayer order from {format_name!r}")
    return m.group(0)


def channel_histograms(bayer, pattern, bit_depth):
    """Per-channel counts of a raw Bayer frame -> {"r": [...], "g", "b"}."""
    hists = {}
    for ch, offs in BAYER_OFFSETS[pattern].items():
        samples = np.concatenate(
            [bayer[dy::2, dx::2].ravel() for dy, dx in offs])
        hists[ch] = np.bincount(samples, minlength=1 << bit_depth).tolist()
    return hists


def process(frame, bayer, crop_limits, scaler_crop,
            threshold=THRESHOLD, min_area=MIN_AREA):
    """Detect blobs in bayer, draw them on frame, and JPEG-encode the frame.
    -> (jpeg_bytes, blobs)"""
    scale, offset = raw_to_frame(bayer.shape[::-1], crop_limits, scaler_crop,
                                 frame.shape[1::-1])
    blobs = detect_blobs(bayer, threshold, min_area, scale, offset)
    ok, jpeg = cv2.imencode(".jpg", annotate(frame, blobs))
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpeg.tobytes(), blobs


def synthetic_capture(frame_size, t):
    """--simulate stand-in for one camera request: a saturated spot circling
    over clipping green speckle on a 16:9 10-bit RGGB sensor, previewed as a
    centred crop scaled to frame_size.
    -> (frame, bayer, crop_limits, scaler_crop)"""
    rw, rh = 1536, 864
    rng = np.random.default_rng(int(t * 10))
    bayer = rng.integers(0, 64, (rh, rw), dtype=np.uint16)
    for green in (bayer[0::2, 1::2], bayer[1::2, 0::2]):
        green[:] = rng.integers(0, 1024, green.shape, dtype=np.uint16)
    cx = int(rw / 2 + rh / 3 * np.cos(t * 0.5))
    cy = int(rh / 2 + rh / 4 * np.sin(t * 0.5))
    cv2.circle(bayer, (cx, cy), rh // 14, 1023, -1)
    cw = rh * frame_size[0] // frame_size[1]
    crop_limits, scaler_crop = (0, 0, rw, rh), ((rw - cw) // 2, 0, cw, rh)
    # OpenCV names RGGB by the 2nd row's pattern: "BayerBG"
    rgb = cv2.cvtColor((bayer >> 2).astype(np.uint8), cv2.COLOR_BayerBG2BGR)
    x = scaler_crop[0]
    frame = cv2.resize(rgb[:, x:x + cw], frame_size,
                       interpolation=cv2.INTER_AREA)
    return frame, bayer, crop_limits, scaler_crop
