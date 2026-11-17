import unittest

import cv2
import numpy as np

from blob_detector import detect_blobs, process


def frame_with_spots(*spots, size=(480, 640)):
    """Green speckle background plus white discs given as (x, y, radius)."""
    rng = np.random.default_rng(0)
    frame = np.zeros((*size, 3), np.uint8)
    frame[..., 1] = rng.integers(0, 256, size, dtype=np.uint8)  # pure green, even clipped
    for x, y, r in spots:
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
    return frame


class DetectBlobsTest(unittest.TestCase):
    def test_finds_spot_centroid(self):
        [blob] = detect_blobs(frame_with_spots((200, 150, 30)))
        self.assertAlmostEqual(blob.x, 200, delta=1)
        self.assertAlmostEqual(blob.y, 150, delta=1)
        self.assertAlmostEqual(blob.radius, 30, delta=2)

    def test_no_spot_means_no_blobs(self):
        self.assertEqual(detect_blobs(frame_with_spots()), [])

    def test_ignores_small_glints_and_sorts_largest_first(self):
        blobs = detect_blobs(frame_with_spots((100, 100, 15), (400, 300, 40),
                                              (500, 80, 3)))
        self.assertEqual([(round(b.x), round(b.y)) for b in blobs],
                         [(400, 300), (100, 100)])

    def test_process_returns_jpeg(self):
        jpeg, blobs = process(frame_with_spots((320, 240, 25)))
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        self.assertEqual(len(blobs), 1)


if __name__ == "__main__":
    unittest.main()
