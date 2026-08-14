"""
Owns one live classroom session: the YOLO26-pose model + tracker, rolling
per-track state (to smooth frame-to-frame flicker into real signals), and
alert generation.

Design choices that matter for accuracy:
- Alerts fire on SUSTAINED conditions (N consecutive seconds), not single
  frames, because a student glancing sideways for half a second is normal,
  not "distracted". This is the single biggest lever against false alerts.
- Every stat the report shows is also written to SQLite as it happens, so
  the report is a replay of real events, not a recomputation from memory.
"""
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

from . import db
from .heuristics import analyze_person

DISTRACTION_SUSTAIN_SECONDS = 8.0   # must be distracted continuously this long before an alert fires
HAND_RAISE_COOLDOWN_SECONDS = 15.0  # don't re-alert for the same raised hand within this window
TRACK_STALE_SECONDS = 5.0           # a track not seen for this long is dropped from "currently visible"

MODEL_WEIGHTS = "weights/yolo26n-pose.pt"


class TrackState:
    def __init__(self):
        self.last_seen = time.time()
        self.focused = None
        self.distracted_since = None
        self.distraction_alerted = False
        self.hand_raised = False
        self.last_hand_alert = 0.0
        self.display_name = None
        self.current_head_state = "unknown"


class SessionEngine:
    def __init__(self, session_id):
        self.session_id = session_id
        self.model = YOLO(MODEL_WEIGHTS)
        self.tracks = {}  # track_id -> TrackState
        self.lock = threading.Lock()
        self.last_boxes = []  # most recent per-person render info, for overlay/report

    def process_frame(self, frame_bgr):
        """Run detection+tracking+heuristics on one BGR frame. Returns a JSON-serializable dict."""
        with self.lock:
            results = self.model.track(
                frame_bgr, persist=True, verbose=False, tracker="bytetrack.yaml", conf=0.4
            )
            r = results[0]
            now = time.time()
            people = []

            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                ids = r.boxes.id.int().tolist()
                xyxy = r.boxes.xyxy.tolist()
                kp_xy = r.keypoints.xy.tolist()
                kp_conf = r.keypoints.conf.tolist() if r.keypoints.conf is not None else None

                for i, track_id in enumerate(ids):
                    box = xyxy[i]
                    xy = kp_xy[i]
                    conf = kp_conf[i] if kp_conf is not None else [1.0] * 17
                    signals = analyze_person(xy, conf)

                    state = self.tracks.get(track_id)
                    if state is None:
                        state = TrackState()
                        self.tracks[track_id] = state
                        db.add_alert(self.session_id, track_id, "student_entered",
                                     f"New student detected (Track #{track_id})")

                    state.last_seen = now
                    state.current_head_state = signals.head_state

                    hand_raise_event = False
                    if signals.confidence_ok:
                        state.focused = signals.focused
                        if signals.focused:
                            state.distracted_since = None
                            state.distraction_alerted = False
                        else:
                            if state.distracted_since is None:
                                state.distracted_since = now
                            elif (now - state.distracted_since >= DISTRACTION_SUSTAIN_SECONDS
                                  and not state.distraction_alerted):
                                state.distraction_alerted = True
                                label = state.display_name or f"Track #{track_id}"
                                db.add_alert(
                                    self.session_id, track_id, "prolonged_distraction",
                                    f"{label} appears distracted ({signals.head_state}) for over "
                                    f"{int(DISTRACTION_SUSTAIN_SECONDS)}s",
                                )

                    # Hand raise: edge-triggered with a cooldown so a held-up hand doesn't spam alerts
                    if signals.hand_raised and not state.hand_raised:
                        if now - state.last_hand_alert > HAND_RAISE_COOLDOWN_SECONDS:
                            hand_raise_event = True
                            state.last_hand_alert = now
                            label = state.display_name or f"Track #{track_id}"
                            db.add_alert(self.session_id, track_id, "hand_raised", f"{label} raised a hand")
                    state.hand_raised = signals.hand_raised

                    db.upsert_track(self.session_id, track_id, signals.focused, hand_raise_event)

                    people.append({
                        "track_id": track_id,
                        "box": box,
                        "display_name": state.display_name,
                        **signals.to_dict(),
                    })

            # Drop stale tracks from the "currently visible" view (they stay in DB history)
            for tid, st in list(self.tracks.items()):
                if now - st.last_seen > TRACK_STALE_SECONDS:
                    del self.tracks[tid]

            db.increment_frames(self.session_id)

            headcount = len(people)
            hands_raised = sum(1 for p in people if p["hand_raised"])
            focused_count = sum(1 for p in people if p["focused"] is True)
            distracted_count = sum(1 for p in people if p["focused"] is False)
            db.add_snapshot(self.session_id, headcount, hands_raised, focused_count, distracted_count)

            self.last_boxes = people

            return {
                "people": people,
                "headcount": headcount,
                "hands_raised": hands_raised,
                "focused_count": focused_count,
                "distracted_count": distracted_count,
                "frame_w": frame_bgr.shape[1],
                "frame_h": frame_bgr.shape[0],
            }

    def tag_track(self, track_id, name):
        with self.lock:
            state = self.tracks.get(track_id)
            if state:
                state.display_name = name
        return db.tag_track(self.session_id, track_id, name)

    def render_annotated(self, frame_bgr, detection_result):
        """Draw boxes/labels on a frame for MJPEG/RTSP streaming mode."""
        for p in detection_result["people"]:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            if p["focused"] is True:
                color = (60, 180, 75)
            elif p["focused"] is False:
                color = (60, 60, 220)
            else:
                color = (160, 160, 160)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = p["display_name"] or f"#{p['track_id']}"
            if p["hand_raised"]:
                label += " ✋"
            tag = f"{label} [{p['head_state']}]"
            cv2.putText(frame_bgr, tag, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_bgr


# ---- Registry of active sessions (in-process; fine for a single-server prototype) ----
_SESSIONS = {}
_REGISTRY_LOCK = threading.Lock()


def create_engine(session_id):
    with _REGISTRY_LOCK:
        engine = SessionEngine(session_id)
        _SESSIONS[session_id] = engine
        return engine


def get_engine(session_id):
    return _SESSIONS.get(session_id)


def remove_engine(session_id):
    with _REGISTRY_LOCK:
        _SESSIONS.pop(session_id, None)


def decode_jpeg_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame
