"""
Unit tests for app/heuristics.py using synthetic keypoints (no model, no image
needed) so the geometry logic itself is verified independent of YOLO.

Coordinate convention matches image coords: y increases DOWNWARD.
Layout of a person standing upright, facing the camera, arms down:
  nose    (100, 50)
  l_eye   (95, 45)   r_eye  (105, 45)
  l_ear   (90, 47)   r_ear  (110, 47)
  l_shoulder (85, 80) r_shoulder (115, 80)
  l_elbow (80, 120)   r_elbow (120, 120)
  l_wrist (78, 160)   r_wrist (122, 160)
  l_hip   (88, 170)   r_hip  (112, 170)
  l_knee/ankle omitted (low conf) -- not needed for these heuristics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.heuristics import analyze_person, NOSE, L_EYE, R_EYE, L_EAR, R_EAR, \
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP

FULL_CONF = 0.9
NO_CONF = 0.05


def base_pose():
    xy = [None] * 17
    conf = [NO_CONF] * 17
    xy[NOSE] = (100, 50)
    xy[L_EYE] = (95, 45)
    xy[R_EYE] = (105, 45)
    xy[L_EAR] = (90, 47)
    xy[R_EAR] = (110, 47)
    xy[L_SHOULDER] = (85, 80)
    xy[R_SHOULDER] = (115, 80)
    xy[L_ELBOW] = (80, 120)
    xy[R_ELBOW] = (120, 120)
    xy[L_WRIST] = (78, 160)
    xy[R_WRIST] = (122, 160)
    xy[L_HIP] = (88, 170)
    xy[R_HIP] = (112, 170)
    xy[13] = (0, 0)
    xy[14] = (0, 0)
    xy[15] = (0, 0)
    xy[16] = (0, 0)
    for idx in [NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER,
                L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP]:
        conf[idx] = FULL_CONF
    return xy, conf


def seated_pose():
    """
    The realistic classroom case: student seated at a desk, camera sees head/
    shoulders/arms but the desk hides hips and legs entirely. This is the
    scenario that was completely broken before — hips were required to
    compute anything, so this pose always came back 'unknown'.
    """
    xy, conf = base_pose()
    conf[L_HIP] = NO_CONF
    conf[R_HIP] = NO_CONF
    conf[13] = NO_CONF
    conf[14] = NO_CONF
    conf[15] = NO_CONF
    conf[16] = NO_CONF
    return xy, conf


def test_neutral_pose_is_forward_focused_no_hand():
    xy, conf = base_pose()
    r = analyze_person(xy, conf)
    assert r.head_state == "forward", r.head_state
    assert r.focused is True
    assert r.hand_raised is False
    print("PASS: neutral pose -> forward, focused, no hand raised")


def test_right_wrist_above_shoulder_raises_hand():
    xy, conf = base_pose()
    xy[R_WRIST] = (122, 20)  # well above right shoulder (y=80)
    r = analyze_person(xy, conf)
    assert r.hand_raised is True
    assert r.hand_side == "right"
    print("PASS: raised right wrist detected as hand_raised=True, side=right")


def test_left_wrist_above_shoulder_raises_hand():
    xy, conf = base_pose()
    xy[L_WRIST] = (78, 15)
    r = analyze_person(xy, conf)
    assert r.hand_raised is True
    assert r.hand_side == "left"
    print("PASS: raised left wrist detected as hand_raised=True, side=left")


def test_wrist_slightly_above_shoulder_but_not_enough_no_raise():
    xy, conf = base_pose()
    # torso length = hip_y(170) - shoulder_y(80) = 90; margin = 0.15*90 = 13.5
    # shoulder_y=80, so threshold to count as raised is y < 80 - 13.5 = 66.5
    xy[R_WRIST] = (122, 75)  # above shoulder but not past margin threshold
    r = analyze_person(xy, conf)
    assert r.hand_raised is False, "wrist barely above shoulder should not count as raised"
    print("PASS: wrist just above shoulder (within margin) correctly NOT flagged")


def test_head_turned_right_via_nose_offset():
    xy, conf = base_pose()
    # Shift nose + eyes far to the right (toward right shoulder / beyond)
    xy[NOSE] = (118, 50)
    xy[L_EYE] = (113, 45)
    xy[R_EYE] = (123, 45)
    conf[L_EAR] = NO_CONF  # left ear occluded when turning right (consistent signal)
    r = analyze_person(xy, conf)
    assert r.head_state in ("turned_right", "turned_left")
    assert r.focused is False
    print(f"PASS: turned head detected as {r.head_state}, focused=False")


def test_head_down():
    xy, conf = base_pose()
    # Drop nose close to shoulder line (chin-to-chest / looking down at desk or phone)
    xy[NOSE] = (100, 75)
    r = analyze_person(xy, conf)
    assert r.head_state == "down"
    assert r.focused is False
    print("PASS: head-down pose detected, focused=False")


def test_low_confidence_keypoints_marked_unknown_not_guessed():
    xy, conf = base_pose()
    conf[NOSE] = NO_CONF
    conf[L_SHOULDER] = NO_CONF
    r = analyze_person(xy, conf)
    assert r.confidence_ok is False
    assert r.head_state == "unknown"
    assert r.focused is None
    print("PASS: insufficient-confidence pose correctly reported as unknown, not guessed")


# ---- The scenarios that were actually broken: seated student, hips hidden by a desk ----

def test_seated_neutral_pose_still_works_without_hips():
    xy, conf = seated_pose()
    r = analyze_person(xy, conf)
    assert r.head_state == "forward", f"expected forward, got {r.head_state} (this was the core bug: hip-dependency broke seated detection)"
    assert r.focused is True
    assert r.hand_raised is False
    print("PASS: seated pose (no hip keypoints) still resolves to forward/focused — core bug fixed")


def test_seated_hand_raised_still_detected_without_hips():
    xy, conf = seated_pose()
    xy[R_WRIST] = (122, 10)
    r = analyze_person(xy, conf)
    assert r.hand_raised is True
    assert r.hand_side == "right"
    print("PASS: seated pose hand-raise detected without hip keypoints")


def test_seated_head_turned_still_detected_without_hips():
    xy, conf = seated_pose()
    xy[NOSE] = (118, 50)
    xy[L_EYE] = (113, 45)
    xy[R_EYE] = (123, 45)
    conf[L_EAR] = NO_CONF
    r = analyze_person(xy, conf)
    assert r.head_state in ("turned_right", "turned_left")
    assert r.focused is False
    print(f"PASS: seated pose head-turn detected without hip keypoints ({r.head_state})")


def test_seated_head_down_still_detected_without_hips():
    xy, conf = seated_pose()
    xy[NOSE] = (100, 75)
    r = analyze_person(xy, conf)
    assert r.head_state == "down"
    assert r.focused is False
    print("PASS: seated pose head-down detected without hip keypoints")


def test_only_shoulders_and_face_visible_at_all_is_enough():
    """Even with elbows/wrists also hidden (arms under the desk), head-direction should still work."""
    xy, conf = seated_pose()
    conf[L_WRIST] = NO_CONF
    conf[R_WRIST] = NO_CONF
    conf[L_ELBOW] = NO_CONF
    conf[R_ELBOW] = NO_CONF
    r = analyze_person(xy, conf)
    assert r.head_state == "forward"
    assert r.focused is True
    assert r.hand_raised is False  # can't confirm raised without wrist visibility -- correctly cautious
    print("PASS: head-direction works even with arms fully hidden under the desk")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} heuristic tests passed.")
