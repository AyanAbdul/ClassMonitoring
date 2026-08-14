"""
Heuristics that turn raw COCO-17 keypoints (from YOLO26-pose) into classroom
signals: hand-raised, head direction, and focused/distracted.

IMPORTANT — these are geometric estimates, not ground truth. They are
confidence-gated (a keypoint below CONF_THRESH is treated as "unknown" and
excluded from the decision) and every result carries enough info for the
caller to know when it was unsure. The UI and reports must present these as
estimates, never as certainties.

COCO-17 keypoint order (what YOLO pose models output):
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12

CONF_THRESH = 0.30
HAND_RAISE_MARGIN_RATIO = 0.15  # wrist must clear shoulder by this fraction of the scale reference
HEAD_TURN_RATIO = 0.55          # normalized nose offset beyond which we call it "turned"
HEAD_DOWN_RATIO = 0.20          # normalized (shoulder_y - nose_y)/scale below which we call it "down"

# When hip keypoints ARE visible (e.g. a wide CCTV shot showing full bodies), we use real
# torso length (shoulder line to hip line) as the scale reference — most accurate option.
# When they AREN'T (the common case: a student seated at a desk, hips hidden below the desk,
# only head/shoulders/arms visible to the camera), we fall back to shoulder width instead,
# converted into torso-equivalent units using average adult/adolescent body proportions.
# This fallback is what makes the app actually work in a real classroom — requiring hip
# keypoints was the bug that made hand-raise/head-direction/focus silently do nothing for
# any seated student.
SHOULDER_WIDTH_TO_TORSO = 1.4


class PersonSignals:
    __slots__ = (
        "hand_raised", "hand_side", "head_state", "focused",
        "confidence_ok", "reason",
    )

    def __init__(self, hand_raised, hand_side, head_state, focused, confidence_ok, reason=""):
        self.hand_raised = hand_raised
        self.hand_side = hand_side
        self.head_state = head_state
        self.focused = focused
        self.confidence_ok = confidence_ok
        self.reason = reason

    def to_dict(self):
        return {
            "hand_raised": self.hand_raised,
            "hand_side": self.hand_side,
            "head_state": self.head_state,
            "focused": self.focused,
            "confidence_ok": self.confidence_ok,
            "reason": self.reason,
        }


def _pt(xy, conf, idx):
    """Return (x, y, ok) for a keypoint, ok=False if below confidence threshold."""
    return xy[idx][0], xy[idx][1], conf[idx] >= CONF_THRESH


def _scale_reference(xy, conf):
    """
    Return real hip-based torso length (shoulder line to hip line) when both
    are visible. Returns None otherwise — the caller falls back to shoulder
    width in that case (see analyze_person), which is the common path for a
    student seated at a desk with hips out of frame.
    """
    _, ls_y, ls_ok = _pt(xy, conf, L_SHOULDER)
    _, rs_y, rs_ok = _pt(xy, conf, R_SHOULDER)
    _, lh_y, lh_ok = _pt(xy, conf, L_HIP)
    _, rh_y, rh_ok = _pt(xy, conf, R_HIP)

    shoulder_ys = [y for y, ok in [(ls_y, ls_ok), (rs_y, rs_ok)] if ok]
    hip_ys = [y for y, ok in [(lh_y, lh_ok), (rh_y, rh_ok)] if ok]

    if shoulder_ys and hip_ys:
        torso = (sum(hip_ys) / len(hip_ys)) - (sum(shoulder_ys) / len(shoulder_ys))
        if torso > 1e-3:
            return torso
    return None


def _shoulder_width(xy, conf):
    lsx, _, ls_ok = _pt(xy, conf, L_SHOULDER)
    rsx, _, rs_ok = _pt(xy, conf, R_SHOULDER)
    if ls_ok and rs_ok:
        w = abs(lsx - rsx)
        if w > 1e-3:
            return w
    return None


def analyze_person(xy, conf):
    """
    xy: list/array of 17 (x, y) pairs
    conf: list/array of 17 confidence floats
    Returns a PersonSignals instance.
    """
    scale = _scale_reference(xy, conf)
    shoulder_w = _shoulder_width(xy, conf)
    if scale is None and shoulder_w is not None:
        scale = shoulder_w * SHOULDER_WIDTH_TO_TORSO
    if scale is None:
        # Truly can't normalize anything (upper body not visible at all)
        return PersonSignals(False, None, "unknown", None, False, "insufficient keypoints (shoulders/torso)")

    # ---- Hand raised ----
    hand_raised = False
    hand_side = None
    for side, wrist_idx, shoulder_idx in (("left", L_WRIST, L_SHOULDER), ("right", R_WRIST, R_SHOULDER)):
        wx, wy, w_ok = _pt(xy, conf, wrist_idx)
        sx, sy, s_ok = _pt(xy, conf, shoulder_idx)
        if w_ok and s_ok:
            if wy < sy - HAND_RAISE_MARGIN_RATIO * scale:
                hand_raised = True
                hand_side = side
                break  # one raised hand is enough to flag the person

    # ---- Head direction / focus ----
    nx, ny, n_ok = _pt(xy, conf, NOSE)
    lsx, lsy, ls_ok = _pt(xy, conf, L_SHOULDER)
    rsx, rsy, rs_ok = _pt(xy, conf, R_SHOULDER)
    _, _, le_ok = _pt(xy, conf, L_EAR)
    _, _, re_ok = _pt(xy, conf, R_EAR)

    if not (n_ok and ls_ok and rs_ok):
        return PersonSignals(hand_raised, hand_side, "unknown", None, False, "insufficient keypoints (face/shoulders)")

    shoulder_cx = (lsx + rsx) / 2.0
    shoulder_w2 = abs(lsx - rsx)
    if shoulder_w2 < 1e-3:
        return PersonSignals(hand_raised, hand_side, "unknown", None, False, "shoulders too close together")

    nose_offset = (nx - shoulder_cx) / (shoulder_w2 / 2.0)  # ~ -1..1 when roughly facing camera
    shoulder_cy = (lsy + rsy) / 2.0
    down_ratio = (shoulder_cy - ny) / scale  # smaller/negative => head dropped down

    if down_ratio < HEAD_DOWN_RATIO:
        head_state = "down"
        focused = False
    elif abs(nose_offset) > HEAD_TURN_RATIO or (le_ok != re_ok):
        # Turned enough, or one ear has disappeared behind the head (classic side-turn signal)
        head_state = "turned_left" if nose_offset < 0 else "turned_right"
        focused = False
    else:
        head_state = "forward"
        focused = True

    return PersonSignals(hand_raised, hand_side, head_state, focused, True, "")
