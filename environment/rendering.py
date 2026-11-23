"""Live 3D rendering bridge: streams environment state to the three.js viewer.

A tiny HTTP server serves `assets/viewer/` and a WebSocket server pushes one
JSON state message per environment step to every connected browser.
"""

import asyncio
import functools
import http.server
import json
import threading
import time
import webbrowser
from pathlib import Path

import websockets

VIEWER_DIR = Path(__file__).resolve().parent.parent / "assets" / "viewer"


class LiveRenderer:
    def __init__(self, http_port=8000, ws_port=8765, open_browser=True, fps=8):
        # Start the HTTP file server and the WebSocket broadcast loop.
        self.ws_port = ws_port
        self.http_port = http_port
        self.fps = fps
        self._clients = set()
        self._last_state = None
        self._loop = None
        self._start_http()
        self._start_ws()
        if open_browser:
            webbrowser.open(f"http://localhost:{http_port}")

    def _start_http(self):
        # Serve the viewer directory on a daemon thread.
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(VIEWER_DIR))
        self._httpd = http.server.ThreadingHTTPServer(("localhost", self.http_port), handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def _start_ws(self):
        # Run the asyncio WebSocket server on its own thread.
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def main():
                async with websockets.serve(self._on_client, "localhost", self.ws_port):
                    ready.set()
                    await asyncio.Future()

            self._loop.run_until_complete(main())

        threading.Thread(target=run, daemon=True).start()
        ready.wait(timeout=5)

    async def _on_client(self, ws):
        # Register a browser tab and replay the latest state to it.
        self._clients.add(ws)
        try:
            if self._last_state:
                await ws.send(self._last_state)
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    def update(self, state):
        # Broadcast one environment snapshot and pace to the target frame rate.
        self._last_state = json.dumps(state)

        async def broadcast():
            for ws in list(self._clients):
                try:
                    await ws.send(self._last_state)
                except websockets.ConnectionClosed:
                    self._clients.discard(ws)

        if self._loop:
            asyncio.run_coroutine_threadsafe(broadcast(), self._loop).result(timeout=1)
        time.sleep(1.0 / self.fps)

    def wait_for_client(self, timeout=60):
        # Block until a browser tab connects so the demo is never missed.
        start = time.time()
        while not self._clients and time.time() - start < timeout:
            time.sleep(0.1)
        return bool(self._clients)
