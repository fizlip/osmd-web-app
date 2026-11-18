import unittest

import cv2
import numpy as np

from blob_detector import (bayer_pattern, channel_histograms, detect_blobs,
                           process, raw_to_frame, synthetic_capture)


def bayer_with_spots(*spots, level=1023, size=(480, 640)):
    """RGGB frame: dim red/blue, green sites clipping at random, plus discs
    where all four samples are `level`, given as (x, y, radius)."""
    rng = np.random.default_rng(0)
    bayer = rng.integers(0, 64, size, dtype=np.uint16)
    for green in (bayer[0::2, 1::2], bayer[1::2, 0::2]):
        green[:] = rng.integers(900, 1024, green.shape, dtype=np.uint16)
    for x, y, r in spots:
        cv2.circle(bayer, (x, y), r, level, -1)
    return bayer


class RawToFrameTest(unittest.TestCase):
    def test_maps_centred_crop_of_wider_sensor(self):
        # 16:9 sensor read 2x binned; ISP shows the centred 4:3 crop at 640x480
        (kx, ky), (ox, oy) = raw_to_frame(
            (2304, 1296), (0, 0, 4608, 2592), (576, 0, 3456, 2592), (640, 480))
        self.assertAlmostEqual(kx, 2 * 640 / 3456)
        self.assertAlmostEqual(ky, 2 * 480 / 2592)
        centre = (2304 / 2 - 0.5, 1296 / 2 - 0.5)  # raw image centre
        self.assertAlmostEqual(centre[0] * kx + ox, 319.5)
        self.assertAlmostEqual(centre[1] * ky + oy, 239.5)
        left_edge = 576 / 2 - 0.5                 # raw x of the crop's edge
        self.assertAlmostEqual(left_edge * kx + ox, -0.5)


class DetectBlobsTest(unittest.TestCase):
    def test_finds_spot_centroid(self):
        [blob] = detect_blobs(bayer_with_spots((200, 150, 30)))
        self.assertAlmostEqual(blob.x, 200, delta=1)
        self.assertAlmostEqual(blob.y, 150, delta=1)
        self.assertAlmostEqual(blob.radius, 30, delta=2)

    def test_clipped_green_speckle_alone_is_not_a_blob(self):
        self.assertEqual(detect_blobs(bayer_with_spots()), [])

    def test_threshold_is_in_10_bit_levels(self):
        spot = bayer_with_spots((200, 150, 30), level=990)
        self.assertEqual(detect_blobs(spot, threshold=1000), [])
        self.assertEqual(len(detect_blobs(spot, threshold=980)), 1)

    def test_ignores_small_glints_and_sorts_largest_first(self):
        blobs = detect_blobs(bayer_with_spots((100, 100, 15), (400, 300, 40),
                                              (500, 80, 3)))
        self.assertEqual([(round(b.x), round(b.y)) for b in blobs],
                         [(400, 300), (100, 100)])

    def test_coordinates_and_area_are_mapped_to_the_frame(self):
        [blob] = detect_blobs(bayer_with_spots((200, 150, 30)),
                              scale=(0.5, 0.5), offset=(10, 20))
        self.assertAlmostEqual(blob.x, 200 * 0.5 + 10, delta=1)
        self.assertAlmostEqual(blob.y, 150 * 0.5 + 20, delta=1)
        self.assertAlmostEqual(blob.radius, 15, delta=1)


class HistogramTest(unittest.TestCase):
    def test_parses_bayer_orders_from_format_names(self):
        for fmt, pattern in [("SRGGB10", "RGGB"), ("SRGGB16", "RGGB"),
                             ("SBGGR10_CSI2P", "BGGR"), ("SGRBG12", "GRBG"),
                             ("SGBRG8", "GBRG"), ("BGGR16_PISP_COMP1", "BGGR")]:
            self.assertEqual(bayer_pattern(fmt), pattern)
        with self.assertRaises(ValueError):
            bayer_pattern("YUV420")

    def test_channels_landed_on_the_right_sites(self):
        bayer = np.zeros((8, 8), np.uint16)
        bayer[0::2, 0::2] = 11   # RGGB: R site
        bayer[0::2, 1::2] = 22   # G1
        bayer[1::2, 0::2] = 33   # G2
        bayer[1::2, 1::2] = 44   # B
        h = channel_histograms(bayer, "RGGB", 10)
        self.assertEqual(len(h["r"]), 1024)
        self.assertEqual(h["r"][11], 16)
        self.assertEqual(h["g"][22] + h["g"][33], 32)  # both greens in g
        self.assertEqual(h["b"][44], 16)


class ProcessTest(unittest.TestCase):
    def test_simulated_capture_end_to_end(self):
        frame, bayer, crop_limits, scaler_crop = synthetic_capture((640, 480), 0)
        jpeg, [blob] = process(frame, bayer, crop_limits, scaler_crop)
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        # spot drawn at raw (1056, 432); preview is the centred 1152x864 crop
        self.assertAlmostEqual(blob.x, (1056 - 192) * 640 / 1152, delta=2)
        self.assertAlmostEqual(blob.y, 432 * 480 / 864, delta=2)


if __name__ == "__main__":
    unittest.main()
