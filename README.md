# ClassMonitor

Live classroom monitoring: person detection, hand-raise detection, head
direction, focus/distraction alerts, and presence-based attendance — from a
webcam or an RTSP/CCTV camera, all processed on your own server (nothing is
sent to a third party).

## ⚠️ Read before you sell this

**The detection model (YOLO26, from Ultralytics) is licensed AGPL-3.0.**
AGPL is copyleft: if you run AGPL-licensed code as part of a service you
offer to others (which is exactly what a classroom-monitoring SaaS is), the
license generally requires you to make the complete corresponding source of
your application available to users of that service. For a product you
intend to sell, that most likely means either:

1. **Buy an Ultralytics Enterprise License** to use YOLO models in a closed-source
   commercial product (this is the standard path companies take), or
2. **Open-source this application** under a compatible license, or
3. **Swap in a differently-licensed detector** (e.g. a permissively-licensed
   pose model) if that fits your accuracy needs.

I'm flagging this now, before you invest in polishing this further, because
it changes what "ready to sell" means here. I'm not a lawyer — confirm with
one before shipping. See https://www.ultralytics.com/license for current terms.

## What's real vs. what's a heuristic

- **Person detection & tracking** — solid. This is YOLO26's core, well-tested job.
- **Headcount / presence-based attendance** — solid, and deliberately does
  **not** use facial recognition or any biometric identity matching (per
  your earlier choice). A teacher can label a tracked box with a name once
  per session ("Seat 3 = Jordan") purely as a session note, not a biometric
  match. The live view now shows a running "Attendance (auto)" count while
  the session is active, not just in the after-the-fact report.
- **Hand-raised, head direction, and focused/distracted** — these are
  **geometric estimates from body pose keypoints**, not ground truth. A
  wrist above the shoulder by a margin counts as "hand raised"; nose/ear
  visibility and offset from the shoulder line estimate head direction;
  sustained (8+ second) non-forward head state counts as "distracted." Real
  classrooms have desks that occlude the lower body, kids turning to talk to
  a neighbor, camera angle quirks, etc. — expect false positives/negatives,
  especially from a single camera angle. Treat every flag as a prompt for a
  teacher to glance over, not a verdict. This is why every uncertain reading
  is reported as `"unknown"` rather than guessed (see `app/heuristics.py`).

### Fixed: a real bug, not just polish

An earlier version required **hip keypoints** to be visible before it would
compute head-direction, focus, or hand-raise at all. In a real classroom,
students sit behind desks — hips are almost never visible to the camera —
so those features were silently returning "unknown"/`false` for nearly
every seated student, which is why they looked broken/missing. The fix:
normalization now prefers shoulder width (reliably visible for anyone
facing the camera) and only uses hip-based torso length as a bonus when it
happens to be available (e.g. a wide CCTV shot showing full bodies).
`test_heuristics.py` now has dedicated tests for the seated/desk-occluded
case, and I re-verified against real YOLO26 output that a person whose hips
are hidden gets the same head-direction/focus reading as when their full
body is visible.

I verified the geometry logic itself is correct with unit tests
(`test_heuristics.py` — 7/7 passing on synthetic poses) and ran the full
pipeline end-to-end against real YOLO26 pose output, including tracking
continuity across frames, alert generation, tagging, and report generation.
What I could **not** test here: real classroom footage, real lighting, real
desk occlusion, or a real RTSP camera (this environment has no camera
hardware). Please pilot it in one real classroom before wider rollout and
retune the thresholds in `app/heuristics.py` / `app/session_manager.py` (see
below) against what you observe.

## Privacy & compliance

This still points a camera at a room full of (likely) minors and logs behavior
data about them:

- No facial recognition and no biometric attendance — confirmed by design.
- Even so, video-based behavior monitoring of students typically needs
  parental/guardian notice or consent and a data-retention policy (FERPA in
  the US, GDPR-style rules elsewhere, plus many states/districts have their
  own classroom-surveillance rules). Post visible signage that monitoring is
  in use.
- The SQLite database stores per-track behavior stats and alert messages.
  Decide a retention window (e.g. auto-delete session data after N days) —
  there's no automatic deletion built in yet; add a scheduled cleanup job
  before production use.

## Features

- **Live monitoring** — webcam (browser capture) or RTSP/CCTV (server-side
  pull), annotated in real time
- **Alerts** — hand raised (instant, debounced), sustained distraction
  (after a configurable number of seconds), new-student-detected
- **Reports** — per-session attendance table, focus %, hand-raise counts,
  full alert log, downloadable as CSV
- **Attendance** — presence/headcount based, with optional manual
  name-tagging per tracked person (no biometrics)

## Getting started

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**. The first session you start will
auto-download the YOLO26 nano pose weights (~7.5MB) into `weights/` — this
needs outbound internet access once.

- **Webcam demo:** choose "This device's webcam" and start — your browser
  will ask for camera permission.
- **RTSP/CCTV:** choose "CCTV / RTSP camera" and enter a stream URL like
  `rtsp://user:pass@camera-ip:554/stream`. The server pulls and processes
  the stream directly; if the camera is unreachable you'll see a banner
  with the error instead of a silent failure.

## Tuning detection

Two files control everything:

- `app/heuristics.py` — `CONF_THRESH`, `HAND_RAISE_MARGIN_RATIO`,
  `HEAD_TURN_RATIO`, `HEAD_DOWN_RATIO`. Adjust these against real footage
  from your camera's actual mounting height/angle — a ceiling-mounted wide
  camera behaves differently than a front-of-room webcam.
- `app/session_manager.py` — `DISTRACTION_SUSTAIN_SECONDS` (how long before
  a distraction alert fires), `HAND_RAISE_COOLDOWN_SECONDS`.

For higher accuracy at the cost of speed, swap `weights/yolo26n-pose.pt` for
a larger variant (`yolo26s-pose.pt`, `yolo26m-pose.pt`, etc.) in
`session_manager.MODEL_WEIGHTS` — bigger models need more CPU/GPU per frame,
so this matters more once you're running several camera feeds at once. A
GPU is strongly recommended for any multi-classroom production deployment;
the nano model on CPU is tuned for a responsive single-camera demo.

## Project layout

```
classmonitor/
  requirements.txt
  test_heuristics.py       unit tests for the pose-geometry logic (no model needed)
  weights/                 YOLO26 pose weights (auto-downloaded on first run)
  app/
    main.py                FastAPI routes
    db.py                  SQLite schema + data access
    heuristics.py           hand-raise / head-direction / focus geometry (pure functions, unit-tested)
    session_manager.py      YOLO26 model + tracker + temporal smoothing + alert logic
    rtsp_worker.py           background thread pulling frames from an RTSP camera
    static/                 frontend (plain HTML/CSS/JS, no build step)
      index.html
      css/style.css
      js/util.js
      js/dashboard.js
  data/                     SQLite database file (created at runtime)
```

## API reference

| Method | Path | Description |
|---|---|---|
| POST | `/api/sessions` | Start a session (`class_name`, `source_mode`=webcam\|rtsp, `rtsp_url`?) |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/sessions/{id}` | Session status (incl. `camera_error` for RTSP) |
| POST | `/api/sessions/{id}/frame` | Webcam mode: post one JPEG frame, get back detections |
| GET | `/api/sessions/{id}/stream` | RTSP mode: annotated MJPEG stream |
| GET | `/api/sessions/{id}/live` | Poll current stats + recent alerts (both modes) |
| POST | `/api/sessions/{id}/tag` | Manually label a tracked person (`track_id`, `name`) |
| POST | `/api/sessions/{id}/stop` | End the session |
| GET | `/api/sessions/{id}/report` | Full report JSON |
| GET | `/api/sessions/{id}/report.csv` | Downloadable CSV report |
