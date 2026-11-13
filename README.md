# Ring Rotator Website

A small web app for the Raspberry Pi that lets a user set a rotation angle in
a browser and relays it to an Arduino over USB serial.

- UI: `index.html` served by `server.py` (Python stdlib HTTP server, no web
  framework) — a real WebGL sphere (Three.js vendored under `libs/`, so no
  internet needed). Drag on the sphere surface to move the target point;
  drag on the background to orbit the camera, pinch/scroll to zoom.
  A Send button commits both angles
- API: `POST /api/rotate` with `{"polar": 30, "azimuth": -45}` writes
  `DET_POL_ABS:30\n` and `DET_AZ_ABS:-45\n` back-to-back so the FreeRTOS
  firmware moves both motors in parallel (either field may be omitted;
  `angle` is a legacy alias for `polar`). The reply is read until each
  command's `<motor> OK` ack (`DET_POL OK`, `DET_AZ OK`) has been seen,
  and the combined text is returned
- The serial port is opened once and kept open (opening `/dev/ttyACM0` toggles
  DTR, which resets the Arduino — re-opening per click would reset it every time)
- Camera: `GET /camera` serves a page showing the Pi Camera via an MJPEG
  stream at `GET /api/stream` (uses `picamera2`; in `--simulate` mode it emits
  a generated test pattern)
- Designed to extend to 4 motors later: add more fields to the request and
  more command strings

## Run on the Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-serial        # or: pip install -r requirements.txt
sudo apt install -y python3-picamera2     # for the /camera page (Pi OS >= Bullseye)
sudo usermod -aG dialout "$USER"          # then log out and back in
python3 server.py
```

Then browse to `http://<pi-address>:8000` from any machine on the network.

Options:

```
--port /dev/ttyACM0   serial device (default)
--baud 38400          must match the Arduino's Serial.begin() rate
--http-port 8000      HTTP listen port
--bind 0.0.0.0        HTTP listen address
--simulate            run with no hardware; returns fake "OK" replies
```

## Things to check on the Arduino side

- **Baud rate**: `--baud` must equal the `Serial.begin(...)` in the sketch
- **Line ending**: the server sends the command followed by `\n`. If the sketch
  expects `\r\n`, change `b"\n"` to `b"\r\n"` in `server.py`
- **Completion acks**: each command is acknowledged with a `<motor> OK`
  line (`DET_POL OK`, `DET_AZ OK`); the server waits for every command it
  sent, in any order. The expected name is derived by stripping `_ABS`
  from the command head

## Run it at boot (systemd)

Create `/etc/systemd/system/ring-rotator.service`:

```ini
[Unit]
Description=Ring rotator web server
After=dev-ttyACM0.device

[Service]
ExecStart=/usr/bin/python3 /home/pi/website/server.py
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl enable --now ring-rotator
```

## Test without hardware

On any machine (e.g. your Mac):

```bash
python3 server.py --simulate
curl -X POST localhost:8000/api/rotate -H 'Content-Type: application/json' -d '{"polar": 90, "azimuth": -45}'
```
