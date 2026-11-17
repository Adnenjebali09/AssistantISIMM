"""Intégration locale. HTTP sur loopback et poids MediaPipe sont nécessaires au test vidéo."""

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import multiprocessing as mp
import os
import queue
import threading
import time
import unittest

from isimm.config import Settings
from isimm.vision import camera_worker


class CameraIntegration(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("ISIMM_TEST_CAMERA") == "1",
        "Activer ISIMM_TEST_CAMERA=1 pour le flux MJPEG de test",
    )
    def test_lost_then_reconnected_stream(self):
        import cv2
        import numpy as np

        ctx = mp.get_context("spawn")
        frames, events = ctx.Queue(1), ctx.Queue()
        stop, armed = ctx.Event(), ctx.Event()
        armed.set()
        active = threading.Event()
        frame = cv2.imencode(".jpg", np.full((180, 320, 3), 128, dtype="uint8"))[
            1
        ].tobytes()

        class Stream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if not active.is_set():
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=part"
                )
                self.end_headers()
                packet = (
                    b"--part\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    while active.is_set() and not stop.is_set():
                        self.wfile.write(packet)
                        time.sleep(0.05)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Stream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cfg = Settings(camera=f"http://127.0.0.1:{server.server_port}/video")
        proc = ctx.Process(
            target=camera_worker, args=(cfg, frames, events, stop, armed)
        )
        proc.start()

        def expect_image(present, timeout=12):
            until = time.monotonic() + timeout
            while time.monotonic() < until:
                try:
                    data, label, progress = frames.get(timeout=0.3)
                except queue.Empty:
                    continue
                if bool(data) == present:
                    if present:
                        self.assertEqual(label, "Aucun visage")
                        self.assertEqual(progress, 0)
                    return
            self.fail("Transition vidéo absente")

        try:
            expect_image(False)
            active.set()
            expect_image(True)
            active.clear()
            expect_image(False)
            active.set()
            expect_image(True)
        finally:
            stop.set()
            active.clear()
            proc.join(4)
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
            server.shutdown()
            server.server_close()
        self.assertEqual(proc.exitcode, 0)
