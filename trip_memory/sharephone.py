"""Send a finished video to a phone on the same Wi-Fi: a QR code that the phone's camera reads opens a download
link. Only that one file is offered, under a random secret link, and only for a while (then the link stops)."""
from __future__ import annotations

import io
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import segno

OPEN_MINUTES = 30
CHUNK = 1 << 20

_lock = threading.Lock()
_state: dict = {"server": None, "files": {}, "until": 0.0}


def lan_ip() -> str | None:
    """This computer's address on the Wi-Fi (None when it is on no network)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.1", 9))  # (no packet is sent: this only picks the network the Mac would use)
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if ip.startswith("127.") else ip


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # (quiet)
        pass

    def do_GET(self) -> None:  # noqa: N802
        token = self.path.strip("/").split("/")[0]
        with _lock:
            path = _state["files"].get(token)
            open_now = time.time() < _state["until"]
        if not path or not open_now or not Path(path).is_file():
            self.send_error(404, "This link has ended. Make a new QR code in Trip Memory.")
            return
        size = Path(path).stat().st_size
        start, end = 0, size - 1
        ranged = self.headers.get("Range", "")
        if ranged.startswith("bytes="):
            a, _, b = ranged[6:].partition("-")
            try:
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if a and b else size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        name = Path(path).name
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    data = f.read(min(CHUNK, left))
                    if not data:
                        break
                    self.wfile.write(data)
                    left -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _stopper() -> None:
    while True:
        time.sleep(30)
        with _lock:
            if time.time() >= _state["until"] and _state["server"]:
                server, _state["server"], _state["files"] = _state["server"], None, {}
                break
    server.shutdown()
    server.server_close()


def share(video: Path, ip: str | None = None) -> str:
    """Offer this video on the Wi-Fi for OPEN_MINUTES; the link to put in the QR code."""
    ip = ip or lan_ip()
    if not ip:
        raise OSError("This Mac is on no Wi-Fi network: join the same Wi-Fi as your phone and try again.")
    token = secrets.token_urlsafe(12)
    with _lock:
        _state["files"][token] = str(video)
        _state["until"] = time.time() + OPEN_MINUTES * 60
        if _state["server"] is None:
            server = ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
            _state["server"] = server
            threading.Thread(target=server.serve_forever, daemon=True).start()
            threading.Thread(target=_stopper, daemon=True).start()
        port = _state["server"].server_address[1]
    return f"http://{ip}:{port}/{token}/{quote(video.name)}"


def qr_svg(text: str) -> str:
    """The QR code as an SVG picture (black on white, big enough to read from a screen)."""
    out = io.BytesIO()
    segno.make(text, error="m").save(out, kind="svg", scale=8, border=3, xmldecl=False, svgns=True)
    return out.getvalue().decode()
