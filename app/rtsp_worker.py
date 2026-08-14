"""
Pulls frames from an RTSP (or any OpenCV-openable) camera URL in a background
thread, runs them through the session's detection pipeline, and keeps the
latest annotated JPEG buffered for the MJPEG /stream endpoint to serve.

Kept as a separate thread per session so a slow/laggy camera never blocks the
FastAPI event loop.
"""
import threading
import time
import cv2

from . import session_manager


PROCESS_EVERY_N_FRAMES = 3  # run the (relatively heavy) pose model on every Nth captured frame


class RtspWorker:
    def __init__(self, session_id, rtsp_url):
        self.session_id = session_id
        self.rtsp_url = rtsp_url
        self.engine = session_manager.get_engine(session_id)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.latest_jpeg = None
        self.error = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            self.error = f"Could not open video source: {self.rtsp_url}"
            return

        frame_idx = 0
        last_result = None
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "Lost connection to camera stream"
                    time.sleep(0.5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
                self.error = None
                frame_idx += 1

                if frame_idx % PROCESS_EVERY_N_FRAMES == 0 or last_result is None:
                    last_result = self.engine.process_frame(frame)

                annotated = self.engine.render_annotated(frame.copy(), last_result)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self._lock:
                        self.latest_jpeg = buf.tobytes()
        finally:
            cap.release()

    def get_jpeg(self):
        with self._lock:
            return self.latest_jpeg


_WORKERS = {}


def start_worker(session_id, rtsp_url):
    worker = RtspWorker(session_id, rtsp_url)
    worker.start()
    _WORKERS[session_id] = worker
    return worker


def get_worker(session_id):
    return _WORKERS.get(session_id)


def stop_worker(session_id):
    worker = _WORKERS.pop(session_id, None)
    if worker:
        worker.stop()
