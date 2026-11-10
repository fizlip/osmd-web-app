#!/usr/bin/env python3
"""HTTP server that relays rotation commands to an Arduino over serial.

Serves a single-page UI plus a small JSON API:

    GET  /              -> index.html
    GET  /camera        -> camera.html (live Pi Camera view)
    GET  /api/stream    -> MJPEG stream (multipart/x-mixed-replace)
    GET  /api/health    -> {"connected": bool, "camera": bool}
    POST /api/rotate    {"polar": 30, "azimuth": -45}
                         -> writes "DET_POL_ABS:30\n" then "DET_AZ_ABS:-45\n"
                         and returns {"ok": bool, "response": "..."}
                         Either field may be omitted; "angle" is a legacy
                         alias for "polar".

The Arduino's reply is accumulated until a line consisting of "OK" is
received (blank lines and text before it are tolerated) or the response
deadline passes.
"""

import argparse
import io
import json
import math
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

RESPONSE_TIMEOUT = 30.0   # seconds to wait for the terminating "OK"
RESET_DELAY = 2.0         # opening the port resets the ATmega via DTR
CAM_SIZE = (640, 480)
CAM_FPS = 10.0
INDEX_HTML = Path(__file__).with_name("index.html")
CAMERA_HTML = Path(__file__).with_name("camera.html")


def format_angle(angle: float) -> str:
    if angle == int(angle):
        return str(int(angle))
    return f"{angle:.6f}".rstrip("0").rstrip(".")


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

    def send(self, command: str, timeout: float = RESPONSE_TIMEOUT):
        """Write command + '\\n', read until an 'OK' line. Returns (ok, text)."""
        with self._lock:
            if self.simulate:
                time.sleep(0.2)
                return True, f"SIMULATED: {command}\nOK"
            try:
                self._ensure_open()
            except Exception as exc:
                return False, f"cannot open {self.port}: {exc}"

            ser = self._ser
            try:
                ser.reset_input_buffer()
                ser.write(command.encode("ascii") + b"\n")
                ser.flush()

                buf = b""
                lines = []
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    chunk = ser.read(ser.in_waiting or 1)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", "replace").strip()
                        if line == "OK":
                            return True, "\n".join([*lines, "OK"])
                        lines.append(line)
                    if buf.strip() == b"OK":  # OK arrived without newline
                        return True, "\n".join([*lines, "OK"])
            except Exception as exc:
                self._ser = None  # force reopen next time (e.g. replugged USB)
                return False, f"serial error: {exc}"

            tail = buf.decode("utf-8", "replace").strip()
            if tail:
                lines.append(tail)
            lines.append(f"[timed out after {timeout:.0f}s waiting for OK]")
            return False, "\n".join(lines)


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

    def __init__(self, simulate=False, size=CAM_SIZE, fps=CAM_FPS):
        self.simulate = simulate
        self.size = size
        self.interval = 1.0 / fps
        self._picam = None
        self._frame = None   # (data, mime)
        self._seq = 0
        self._cond = threading.Condition()
        self._thread = None

    @property
    def available(self):
        return self.simulate or self._picam is not None

    def start(self):
        """Begin capturing. Raises if no camera can be opened."""
        if self.simulate:
            self._ensure_thread(self._simulate_loop)
            return
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        if self._picam is None:
            self._picam = Picamera2()
            self._picam.configure(
                self._picam.create_video_configuration(main={"size": self.size}))
            self._picam.start()
        self._ensure_thread(self._capture_loop)

    def _ensure_thread(self, target):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()

    def _capture_loop(self):
        buf = io.BytesIO()
        while True:
            try:
                buf.seek(0)
                buf.truncate(0)
                self._picam.capture_file(buf, format="jpeg")
            except Exception:
                time.sleep(1)
                continue
            self._publish(buf.getvalue(), "image/jpeg")
            time.sleep(self.interval)

    def _simulate_loop(self):
        while True:
            self._publish(make_png(320, 240, time.monotonic()), "image/png")
            time.sleep(0.4)

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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
        elif self.path == "/camera":
            self._send_html(CAMERA_HTML)
        elif self.path == "/api/stream":
            self._send_stream()
        elif self.path == "/api/health":
            self._send_json(200, {"connected": self.manager.connected,
                                  "camera": self.camera.available})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/rotate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "body must be JSON"})
            return

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

        ok_all, parts = True, []
        for cmd in commands:
            ok, text = self.manager.send(cmd)
            ok_all &= ok
            parts.append(text)
        self._send_json(200, {"ok": ok_all, "response": "\n".join(parts)})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0", help="serial device")
    ap.add_argument("--baud", type=int, default=9600,
                    help="must match the Arduino's Serial.begin()")
    ap.add_argument("--bind", default="0.0.0.0", help="HTTP listen address")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--simulate", action="store_true",
                    help="run without serial hardware; returns fake OK replies")
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

    cam = CameraManager(simulate=args.simulate)
    if not args.simulate and Picamera2 is None:
        print("warning: picamera2 not installed; /camera will return 503")

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
