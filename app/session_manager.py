"""
ClassMonitor Session Engine
YOLO26 Pose + CUDA/CPU + ByteTrack + Classroom Heuristics

Pipeline:

Webcam / RTSP
      ↓
OpenCV frame
      ↓
YOLO26n-Pose
      ↓
ByteTrack
      ↓
COCO-17 keypoints
      ↓
Classroom heuristics
      ├── Hand raised
      ├── Head direction
      └── Focus / distraction
      ↓
SQLite attendance + alerts
      ↓
FastAPI dashboard
"""

import time
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from . import db
from .heuristics import analyze_person


# ============================================================
# CONFIGURATION
# ============================================================

# YOLO26 pose model
MODEL_WEIGHTS = "weights/yolo26n-pose.pt"

# Detection confidence
DETECTION_CONFIDENCE = 0.40

# Sustained distraction threshold
# A student must appear distracted continuously for this duration
# before an alert is generated.
DISTRACTION_SUSTAIN_SECONDS = 8.0

# Prevent repeated hand-raise alerts
HAND_RAISE_COOLDOWN_SECONDS = 15.0

# Remove a tracking state after this much time without detection
TRACK_STALE_SECONDS = 5.0

# ByteTrack configuration
TRACKER_CONFIG = "bytetrack.yaml"


# ============================================================
# TRACK STATE
# ============================================================

class TrackState:
    """
    Stores information about one tracked student/person.
    """

    def __init__(self):
        self.last_seen = time.time()

        # Focus state
        self.focused = None

        # Distraction timing
        self.distracted_since = None
        self.distraction_alerted = False

        # Hand state
        self.hand_raised = False
        self.last_hand_alert = 0.0

        # Optional manually assigned name
        self.display_name = None

        # Head direction
        self.current_head_state = "unknown"


# ============================================================
# SESSION ENGINE
# ============================================================

class SessionEngine:

    def __init__(self, session_id):

        self.session_id = session_id

        # ----------------------------------------------------
        # Select GPU automatically
        # ----------------------------------------------------

        if torch.cuda.is_available():
            self.device = "cuda:0"

            print("=" * 60)
            print("[ClassMonitor] CUDA GPU detected")
            print(
                f"[ClassMonitor] GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )
            print(
                f"[ClassMonitor] CUDA build: "
                f"{torch.version.cuda}"
            )
            print("=" * 60)

        else:
            self.device = "cpu"

            print("=" * 60)
            print("[ClassMonitor] CUDA unavailable")
            print("[ClassMonitor] Falling back to CPU")
            print("=" * 60)

        # ----------------------------------------------------
        # Load YOLO26 once
        # ----------------------------------------------------

        print(
            f"[ClassMonitor] Loading model: "
            f"{MODEL_WEIGHTS}"
        )

        self.model = YOLO(MODEL_WEIGHTS)

        print(
            f"[ClassMonitor] Model task: "
            f"{self.model.task}"
        )

        print(
            f"[ClassMonitor] Inference device: "
            f"{self.device}"
        )

        print("[ClassMonitor] YOLO26 ready")

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.tracks = {}

        # Protect model + tracking state
        self.lock = threading.Lock()

        # Latest detection result
        self.last_boxes = []

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame_bgr):
        """
        Process one BGR OpenCV frame.

        Returns a JSON-serializable dictionary containing:

        - people
        - headcount
        - hands raised
        - focused count
        - distracted count
        - frame dimensions
        """

        if frame_bgr is None:
            return {
                "people": [],
                "headcount": 0,
                "hands_raised": 0,
                "focused_count": 0,
                "distracted_count": 0,
                "frame_w": 0,
                "frame_h": 0,
            }

        with self.lock:

            # ------------------------------------------------
            # YOLO26 Pose + ByteTrack
            # ------------------------------------------------

            results = self.model.track(
                source=frame_bgr,
                persist=True,
                verbose=False,
                tracker=TRACKER_CONFIG,
                conf=DETECTION_CONFIDENCE,
                device=self.device,
            )

            r = results[0]

            now = time.time()

            people = []

            # ------------------------------------------------
            # Extract detections
            # ------------------------------------------------

            if (
                r.boxes is not None
                and r.boxes.id is not None
                and r.keypoints is not None
            ):

                ids = r.boxes.id.int().tolist()

                xyxy = r.boxes.xyxy.tolist()

                kp_xy = r.keypoints.xy.tolist()

                if r.keypoints.conf is not None:
                    kp_conf = r.keypoints.conf.tolist()
                else:
                    kp_conf = None

                # ------------------------------------------------
                # Process every tracked person
                # ------------------------------------------------

                for i, track_id in enumerate(ids):

                    box = xyxy[i]

                    xy = kp_xy[i]

                    if kp_conf is not None:
                        conf = kp_conf[i]
                    else:
                        conf = [1.0] * 17

                    # --------------------------------------------
                    # Classroom heuristics
                    # --------------------------------------------

                    signals = analyze_person(
                        xy,
                        conf
                    )

                    # --------------------------------------------
                    # Get/create track state
                    # --------------------------------------------

                    state = self.tracks.get(track_id)

                    if state is None:

                        state = TrackState()

                        self.tracks[track_id] = state

                        db.add_alert(
                            self.session_id,
                            track_id,
                            "student_entered",
                            (
                                f"New student detected "
                                f"(Track #{track_id})"
                            ),
                        )

                    state.last_seen = now

                    state.current_head_state = (
                        signals.head_state
                    )

                    # ====================================================
                    # FOCUS / DISTRACTION
                    # ====================================================

                    if signals.confidence_ok:

                        state.focused = signals.focused

                        # --------------------------------------------
                        # Student is focused
                        # --------------------------------------------

                        if signals.focused:

                            state.distracted_since = None

                            state.distraction_alerted = False

                        # --------------------------------------------
                        # Student is distracted
                        # --------------------------------------------

                        else:

                            if state.distracted_since is None:

                                state.distracted_since = now

                            else:

                                distraction_duration = (
                                    now
                                    - state.distracted_since
                                )

                                if (
                                    distraction_duration
                                    >= DISTRACTION_SUSTAIN_SECONDS
                                    and not state.distraction_alerted
                                ):

                                    state.distraction_alerted = True

                                    label = (
                                        state.display_name
                                        or f"Track #{track_id}"
                                    )

                                    db.add_alert(
                                        self.session_id,
                                        track_id,
                                        "prolonged_distraction",
                                        (
                                            f"{label} appears distracted "
                                            f"({signals.head_state}) "
                                            f"for over "
                                            f"{int(DISTRACTION_SUSTAIN_SECONDS)}s"
                                        ),
                                    )

                    # ====================================================
                    # HAND RAISE
                    # ====================================================

                    hand_raise_event = False

                    if signals.hand_raised:

                        # Edge-triggered alert
                        if not state.hand_raised:

                            if (
                                now
                                - state.last_hand_alert
                                > HAND_RAISE_COOLDOWN_SECONDS
                            ):

                                hand_raise_event = True

                                state.last_hand_alert = now

                                label = (
                                    state.display_name
                                    or f"Track #{track_id}"
                                )

                                db.add_alert(
                                    self.session_id,
                                    track_id,
                                    "hand_raised",
                                    (
                                        f"{label} raised a hand"
                                    ),
                                )

                    state.hand_raised = (
                        signals.hand_raised
                    )

                    # ====================================================
                    # DATABASE TRACK UPDATE
                    # ====================================================

                    db.upsert_track(
                        self.session_id,
                        track_id,
                        signals.focused,
                        hand_raise_event,
                    )

                    # ====================================================
                    # RESULT FOR THIS PERSON
                    # ====================================================

                    people.append(
                        {
                            "track_id": track_id,

                            "box": box,

                            "display_name":
                                state.display_name,

                            **signals.to_dict(),
                        }
                    )

            # ========================================================
            # REMOVE STALE TRACKS
            # ========================================================

            for tid, state in list(
                self.tracks.items()
            ):

                if (
                    now - state.last_seen
                    > TRACK_STALE_SECONDS
                ):

                    del self.tracks[tid]

            # ========================================================
            # CLASSROOM STATISTICS
            # ========================================================

            db.increment_frames(
                self.session_id
            )

            headcount = len(people)

            hands_raised = sum(
                1
                for p in people
                if p["hand_raised"]
            )

            focused_count = sum(
                1
                for p in people
                if p["focused"] is True
            )

            distracted_count = sum(
                1
                for p in people
                if p["focused"] is False
            )

            # ========================================================
            # SAVE SNAPSHOT
            # ========================================================

            db.add_snapshot(
                self.session_id,
                headcount,
                hands_raised,
                focused_count,
                distracted_count,
            )

            # ========================================================
            # STORE LATEST RESULT
            # ========================================================

            self.last_boxes = people

            # ========================================================
            # RETURN RESULT
            # ========================================================

            return {
                "people": people,

                "headcount": headcount,

                "hands_raised": hands_raised,

                "focused_count": focused_count,

                "distracted_count": distracted_count,

                "frame_w": frame_bgr.shape[1],

                "frame_h": frame_bgr.shape[0],
            }

    # ========================================================
    # MANUAL TRACK TAGGING
    # ========================================================

    def tag_track(
        self,
        track_id,
        name
    ):

        with self.lock:

            state = self.tracks.get(
                track_id
            )

            if state:

                state.display_name = name

        return db.tag_track(
            self.session_id,
            track_id,
            name,
        )

    # ========================================================
    # RENDER ANNOTATED FRAME
    # ========================================================

    def render_annotated(
        self,
        frame_bgr,
        detection_result,
    ):
        """
        Draw bounding boxes and labels.
        """

        for p in detection_result["people"]:

            x1, y1, x2, y2 = [
                int(v)
                for v in p["box"]
            ]

            # --------------------------------------------
            # Color based on focus state
            # --------------------------------------------

            if p["focused"] is True:

                color = (
                    60,
                    180,
                    75,
                )

            elif p["focused"] is False:

                color = (
                    60,
                    60,
                    220,
                )

            else:

                color = (
                    160,
                    160,
                    160,
                )

            # --------------------------------------------
            # Bounding box
            # --------------------------------------------

            cv2.rectangle(
                frame_bgr,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # --------------------------------------------
            # Student label
            # --------------------------------------------

            label = (
                p["display_name"]
                or f"#{p['track_id']}"
            )

            if p["hand_raised"]:

                label += " HAND"

            # --------------------------------------------
            # Head state
            # --------------------------------------------

            tag = (
                f"{label} "
                f"[{p['head_state']}]"
            )

            # --------------------------------------------
            # Draw text
            # --------------------------------------------

            cv2.putText(
                frame_bgr,
                tag,
                (
                    x1,
                    max(0, y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame_bgr


# ============================================================
# ACTIVE SESSION REGISTRY
# ============================================================

_SESSIONS = {}

_REGISTRY_LOCK = threading.Lock()


def create_engine(session_id):

    with _REGISTRY_LOCK:

        engine = SessionEngine(
            session_id
        )

        _SESSIONS[session_id] = engine

        return engine


def get_engine(session_id):

    return _SESSIONS.get(
        session_id
    )


def remove_engine(session_id):

    with _REGISTRY_LOCK:

        _SESSIONS.pop(
            session_id,
            None,
        )


# ============================================================
# JPEG DECODER
# ============================================================

def decode_jpeg_bytes(
    data: bytes
):

    arr = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR,
    )

    return frame