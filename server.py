#!/usr/bin/env python3
"""HTTP server that relays rotation commands to an Arduino over serial.

Serves a single-page UI plus a small JSON API:

    GET  /              -> index.html
    GET  /camera        -> camera.html (live Pi Camera view)
    GET  /api/stream    -> MJPEG stream (multipart/x-mixed-replace) with
                           detected bright blobs and their (x, y) drawn on
    GET  /api/health    -> {"connected": bool, "camera": bool}
    POST /api/rotate    {"polar": 30, "azimuth": -45}
                         -> writes "DET_POL_ABS:30\n" then "DET_AZ_ABS:-45\n"
                         and returns {"ok": bool, "response": "..."}
                         Either field may be omitted; "angle" is a legacy
                         alias for "polar".

Commands are written back-to-back so the FreeRTOS firmware can move the
motors in parallel; the reply is accumulated until each command's
"<MOTOR> OK" completion line (e.g. "DET_POL OK") has been seen, in
whatever order they arrive, or the response deadline passes.
"""

import argparse
import io
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None  # --simulate still works without pyserial installed

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # /api/stream still serves simulated frames in --simulate

try:
    import numpy as np
    import blob_detector
except ImportError:
    blob_detector = None  # no OpenCV/numpy: stream runs without blob overlay

RESPONSE_TIMEOUT = 30.0   # seconds to wait for the terminating "OK"
RESET_DELAY = 2.0         # opening the port resets the ATmega via DTR
CAM_SIZE = (640, 480)
CAM_FPS = 10.0
INDEX_HTML = Path(__file__).with_name("index.html")
CAMERA_HTML = Path(__file__).with_name("camera.html")
LIBS_DIR = Path(__file__).with_name("libs")


def format_angle(angle: float) -> str:
    if angle == int(angle):
        return str(int(angle))
    return f"{angle:.6f}".rstrip("0").rstrip(".")


def expected_ack(command: str) -> str:
    """Completion line the firmware sends, e.g. DET_POL_ABS:30 -> 'DET_POL OK'."""
    head = command.split(":", 1)[0]
    return head.removesuffix("_ABS") + " OK"


class SerialManager:
    """Owns the one serial connection; serializes access across requests."""

    def __init__(self, port: str, baud: int, simulate: bool = False):
        self.port = port
        self.baud = baud
        self.simulate = simulate
        self._ser = None
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self.simulate or bool(self._ser and self._ser.is_open)

    def _ensure_open(self):
        if self.simulate or (self._ser and self._ser.is_open):
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self._ser = serial.Serial(self.port, self.baud, timeout=0.25)
        time.sleep(RESET_DELAY)          # let the Arduino finish rebooting
        self._ser.reset_input_buffer()   # drop bootloader chatter

    def send(self, commands, timeout: float = RESPONSE_TIMEOUT):
        """Write all commands back-to-back (FreeRTOS runs the motors in
        parallel), then read until every command's '<MOTOR> OK' ack has been
        seen, in whatever order they arrive. Returns (all_acked, text)."""
        if isinstance(commands, str):
            commands = [commands]
        acks = [expected_ack(c) for c in commands]
        with self._lock:
            if self.simulate:
                time.sleep(0.2)
                lines = [f"SIMULATED: {c}" for c in commands] + acks
                return True, "\n".join(lines)
            try:
                self._ensure_open()
            except Exception as exc:
                return False, f"cannot open {self.port}: {exc}"

            ser = self._ser
            try:
                ser.reset_input_buffer()
                for cmd in commands:
                    ser.write(cmd.encode("ascii") + b"\n")
                ser.flush()

                pending = set(acks)
                buf = b""
                lines = []
                deadline = time.monotonic() + timeout
                while pending and time.monotonic() < deadline:
                    chunk = ser.read(ser.in_waiting or 1)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", "replace").strip()
                        pending.discard(line)
                        lines.append(line)
                    tail = buf.strip()  # final ack may arrive without newline
                    if tail in pending:
                        lines.append(tail.decode("utf-8", "replace"))
                        pending.discard(tail)
                        buf = b""
            except Exception as exc:
                self._ser = None  # force reopen next time (e.g. replugged USB)
                return False, f"serial error: {exc}"

            tail = buf.decode("utf-8", "replace").strip()
            if tail:
                lines.append(tail)
            if pending:
                lines.append(f"[timed out after {timeout:.0f}s waiting for: "
                             f"{', '.join(sorted(pending))}]")
                return False, "\n".join(lines)
            return True, "\n".join(lines)


def make_png(w, h, t):
    """Minimal truecolour PNG (stdlib only) — used for simulated camera frames."""
    import struct
    import zlib
    bar = int(t * 0.3 % 1 * w)
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter: none
        for x in range(w):
            if abs(x - bar) < 2:
                raw += b"\x00\x00\x00"
            else:
                raw += bytes((x * 255 // w, y * 255 // h, 160))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 1))
            + chunk(b"IEND", b""))


class CameraManager:
    """Captures frames in a background thread; stream clients get the latest."""

    def __init__(self, simulate=False, size=CAM_SIZE, fps=CAM_FPS,
                 blob_args=None):
        self.simulate = simulate
        self.size = size
        self.interval = 1.0 / fps
        self.blob_args = blob_args or {}  # detect_blobs() keyword overrides
        self._picam = None
        self._frame = None   # (data, mime)
        self._seq = 0
        self._cond = threading.Condition()
        self._thread = None
        self._ctrl = {"exposure_us": None, "gain": None, "ae": True}
        self._bayer = None            # latest raw frame (sensor levels)
        self._bayer_pattern = "RGGB"  # synthetic capture is RGGB
        self._bit_depth = 10

    @property
    def available(self):
        return self.simulate or self._picam is not None

    def camera_settings(self):
        """Current exposure controls, e.g. for the /camera page sliders."""
        return dict(self._ctrl)

    def set_camera_settings(self, settings):
        """Update exposure controls; remembered for when the camera opens.
        Keys: ae (bool), exposure_us (int), gain (float)."""
        self._ctrl.update(settings)
        if self._picam is None:
            return
        if self._ctrl["ae"]:
            self._picam.set_controls({"AeEnable": True})
            return
        controls = {"AeEnable": False}
        if self._ctrl["exposure_us"] is not None:
            controls["ExposureTime"] = self._ctrl["exposure_us"]
        if self._ctrl["gain"] is not None:
            controls["AnalogueGain"] = self._ctrl["gain"]
        self._picam.set_controls(controls)

    def start(self):
        """Begin capturing. Raises if no camera can be opened."""
        if self.simulate:
            self._ensure_thread(self._simulate_loop)
            return
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        if self._picam is None:
            self._picam = Picamera2()
            mode = self._full_fov_mode()
            # RGB888 arrays come out in B, G, R order, as OpenCV expects;
            # the unpacked raw format gives one 16-bit word per sensor sample
            self._picam.configure(self._picam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                raw={"size": mode["size"], "format": mode["unpacked"]}))
            self._crop_limits = mode["crop_limits"]
            cfg = self._picam.camera_configuration()
            self._raw_width = cfg["raw"]["size"][0]
            # Pi 4 right-aligns e.g. 10-bit samples (0-1023); Pi 5 left-shifts
            # them to fill 16 bits ("SRGGB16"). Shift back to sensor levels.
            # (older picamera2 lacks cfg["sensor"]; the mode knows it too)
            container = int(re.search(r"\d+$", cfg["raw"]["format"]).group())
            self._raw_shift = container - mode["bit_depth"]
            self._bit_depth = mode["bit_depth"]
            self._bayer_pattern = blob_detector.bayer_pattern(
                cfg["raw"]["format"])
            self._picam.start()
            self.set_camera_settings({})  # apply any controls set pre-start
            if "AfMode" in self._picam.camera_controls:
                from libcamera import controls
                self._picam.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfRange": controls.AfRangeEnum.Full})
        self._ensure_thread(self._capture_loop)

    def _full_fov_mode(self):
        """Smallest sensor mode that reads the whole sensor (no crop), so the
        ISP downscales to self.size instead of using a cropped, zoomed mode."""
        def area(m):
            return m["crop_limits"][2] * m["crop_limits"][3]
        modes = self._picam.sensor_modes
        widest = max(area(m) for m in modes)
        return min((m for m in modes if area(m) == widest),
                   key=lambda m: m["size"][0] * m["size"][1])

    def _ensure_thread(self, target):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()

    def _capture_loop(self):
        buf = io.BytesIO()
        while True:
            try:
                if blob_detector is None:
                    buf.seek(0)
                    buf.truncate(0)
                    self._picam.capture_file(buf, format="jpeg")
                    data = buf.getvalue()
                else:
                    data = self._capture_and_detect()
            except Exception:
                time.sleep(1)
                continue
            self._publish(data, "image/jpeg")
            time.sleep(self.interval)

    def _simulate_loop(self):
        while True:
            t = time.monotonic()
            if blob_detector is None:
                self._publish(make_png(320, 240, t), "image/png")
            else:
                capture = blob_detector.synthetic_capture(self.size, t)
                self._bayer = capture[1]
                self._publish(self._detect(*capture), "image/jpeg")
            time.sleep(0.4)

    def _capture_and_detect(self):
        """Grab the preview and raw frames of one request, so the blobs found
        in the raw data line up with the image they are drawn on."""
        request = self._picam.capture_request()
        try:
            frame = request.make_array("main")
            raw = request.make_array("raw").view(np.uint16)
            scaler_crop = request.get_metadata()["ScalerCrop"]
        finally:
            request.release()
        bayer = raw[:, :self._raw_width] >> self._raw_shift  # drop row padding
        self._bayer = bayer  # new array owned by us; safe after release()
        return self._detect(frame, bayer, self._crop_limits, scaler_crop)

    def histogram(self):
        """R/G/B channel histograms of the latest raw frame, or None."""
        if self._bayer is None or blob_detector is None:
            return None
        return {"bit_depth": self._bit_depth,
                "pattern": self._bayer_pattern,
                **blob_detector.channel_histograms(
                    self._bayer, self._bayer_pattern, self._bit_depth)}

    def _detect(self, frame, bayer, crop_limits, scaler_crop):
        """JPEG of frame with each blob and its (x, y) drawn on it."""
        return blob_detector.process(frame, bayer, crop_limits, scaler_crop,
                                     **self.blob_args)[0]

    def _publish(self, data, mime):
        with self._cond:
            self._frame = (data, mime)
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, after_seq, timeout=10.0):
        """Block until a frame newer than after_seq exists. -> (data, mime, seq)"""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                return None, None, after_seq
            return *self._frame, self._seq


class Handler(BaseHTTPRequestHandler):
    manager: SerialManager = None  # set in main()
    camera: CameraManager = None   # set in main()

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self):
        try:
            self.camera.start()
        except Exception as exc:
            self._send_json(503, {"error": f"camera unavailable: {exc}"})
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        seq = 0
        try:
            while True:
                data, mime, seq = self.camera.wait_frame(seq)
                if data is None:
                    break  # camera stopped producing frames
                self.wfile.write(
                    b"--frame\r\nContent-Type: " + mime.encode()
                    + b"\r\nContent-Length: " + str(len(data)).encode()
                    + b"\r\n\r\n" + data + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected

    def _send_lib(self, name: str):
        if "/" in name or ".." in name or not name.endswith(".js"):
            self._send_json(404, {"error": "not found"})
            return
        f = LIBS_DIR / name
        if not f.is_file():
            self._send_json(404, {"error": "not found"})
            return
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
        elif self.path == "/camera":
            self._send_html(CAMERA_HTML)
        elif self.path.startswith("/libs/"):
            self._send_lib(self.path.removeprefix("/libs/"))
        elif self.path == "/api/stream":
            self._send_stream()
        elif self.path == "/api/health":
            self._send_json(200, {"connected": self.manager.connected,
                                  "camera": self.camera.available})
        elif self.path == "/api/camera-settings":
            s = self.camera.camera_settings()
            s["available"] = self.camera.available
            self._send_json(200, s)
        elif self.path == "/api/histogram":
            hist = self.camera.histogram()
            if hist is None:
                self._send_json(503, {"error": "no raw frame captured yet"})
            else:
                self._send_json(200, hist)
        else:
            self._send_json(404, {"error": "not found"})

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "body must be JSON"})
            return None

    def _camera_settings(self, data):
        try:
            settings = {}
            if "ae" in data:
                settings["ae"] = bool(data["ae"])
            if "exposure_us" in data:
                v = data["exposure_us"]
                if isinstance(v, bool) or not (0 < float(v) <= 1e6):
                    raise ValueError("exposure_us")
                settings["exposure_us"] = int(v)
            if "gain" in data:
                v = data["gain"]
                if isinstance(v, bool) or not (0 < float(v) <= 128):
                    raise ValueError("gain")
                settings["gain"] = float(v)
        except (ValueError, TypeError):
            self._send_json(400, {"error": "settings must be numbers in range"})
            return
        if not settings:
            self._send_json(400, {"error": "expected 'ae', 'exposure_us' "
                                          "and/or 'gain'"})
            return
        if not self.camera.available:
            self._send_json(503, {"error": "camera unavailable"})
            return
        self.camera.set_camera_settings(settings)
        self._send_json(200, {"ok": True})

    def do_POST(self):
        data = self._read_json()
        if data is None:
            return
        if self.path == "/api/camera-settings":
            self._camera_settings(data)
        elif self.path == "/api/rotate":
            self._rotate(data)
        else:
            self._send_json(404, {"error": "not found"})

    def _rotate(self, data):

        def angle_of(key):
            v = data.get(key)
            if v is None:
                return None
            a = float(v)  # raises TypeError/ValueError on bad input
            if isinstance(v, bool) or not math.isfinite(a):
                raise ValueError(key)
            return a

        try:
            polar = angle_of("polar")
            if polar is None:
                polar = angle_of("angle")  # legacy single-axis clients
            azimuth = angle_of("azimuth")
        except (ValueError, TypeError):
            self._send_json(400, {"error": "angles must be finite numbers"})
            return

        commands = []
        if polar is not None:
            commands.append(f"DET_POL_ABS:{format_angle(polar)}")
        if azimuth is not None:
            commands.append(f"DET_AZ_ABS:{format_angle(azimuth)}")
        if not commands:
            self._send_json(400, {"error": "expected 'polar' and/or 'azimuth'"})
            return

        ok, response = self.manager.send(commands)
        self._send_json(200, {"ok": ok, "response": response})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0", help="serial device")
    ap.add_argument("--baud", type=int, default=38400,
                    help="must match the Arduino's Serial.begin()")
    ap.add_argument("--bind", default="0.0.0.0", help="HTTP listen address")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--simulate", action="store_true",
                    help="run without serial hardware; returns fake OK replies")
    ap.add_argument("--blob-threshold", type=int,
                    help="raw sensor level (0-1023 at 10 bits) all four "
                         "samples of a Bayer cell must reach (default 1000)")
    ap.add_argument("--blob-min-area", type=int,
                    help="smallest blob in pixels (default 100)")
    args = ap.parse_args()

    mgr = SerialManager(args.port, args.baud, args.simulate)
    if mgr.simulate:
        print("SIMULATE mode: no serial port will be opened")
    else:
        try:
            mgr._ensure_open()
            print(f"Opened {args.port} @ {args.baud} baud")
        except Exception as exc:
            print(f"warning: {exc} (will retry on the first command)")

    blob_args = {"threshold": args.blob_threshold,
                 "min_area": args.blob_min_area}
    cam = CameraManager(simulate=args.simulate, blob_args={
        k: v for k, v in blob_args.items() if v is not None})
    if not args.simulate and Picamera2 is None:
        print("warning: picamera2 not installed; /camera will return 503")
    if blob_detector is None:
        print("warning: OpenCV/numpy not installed; no blob detection")

    Handler.manager = mgr
    Handler.camera = cam
    server = ThreadingHTTPServer((args.bind, args.http_port), Handler)
    print(f"Serving UI on http://{args.bind}:{args.http_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
