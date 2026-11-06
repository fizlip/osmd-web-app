#!/usr/bin/env python3
"""HTTP server that relays rotation commands to an Arduino over serial.

Serves a single-page UI plus a small JSON API:

    GET  /              -> index.html
    GET  /api/health    -> {"connected": bool}
    POST /api/rotate    {"angle": 123.4} -> writes "DET_POL_ABS:123.4\n"
                         and returns {"ok": bool, "response": "..."}

The Arduino's reply is accumulated until a line consisting of "OK" is
received (blank lines and text before it are tolerated) or the response
deadline passes.
"""

import argparse
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

RESPONSE_TIMEOUT = 30.0   # seconds to wait for the terminating "OK"
RESET_DELAY = 2.0         # opening the port resets the ATmega via DTR
INDEX_HTML = Path(__file__).with_name("index.html")


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


class Handler(BaseHTTPRequestHandler):
    manager: SerialManager = None  # set in main()

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/health":
            self._send_json(200, {"connected": self.manager.connected})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/rotate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            angle = float(data["angle"])
            if isinstance(data["angle"], bool) or not math.isfinite(angle):
                raise ValueError
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._send_json(400, {"error": "body must be JSON with a numeric 'angle'"})
            return

        ok, response = self.manager.send(f"DET_POL_ABS:{format_angle(angle)}")
        self._send_json(200, {"ok": ok, "response": response})


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

    Handler.manager = mgr
    server = ThreadingHTTPServer((args.bind, args.http_port), Handler)
    print(f"Serving UI on http://{args.bind}:{args.http_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
