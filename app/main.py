import uuid
import io
import csv
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, session_manager, rtsp_worker

app = FastAPI(title="ClassMonitor")


@app.on_event("startup")
def startup():
    db.init_db()


# ---------------- Session lifecycle ----------------

@app.post("/api/sessions")
def create_session(class_name: str = Form(...), source_mode: str = Form(...), rtsp_url: str = Form(None)):
    if source_mode not in ("webcam", "rtsp"):
        raise HTTPException(400, "source_mode must be 'webcam' or 'rtsp'")
    if source_mode == "rtsp" and not rtsp_url:
        raise HTTPException(400, "rtsp_url is required when source_mode is 'rtsp'")
    if not class_name or not class_name.strip():
        raise HTTPException(400, "class_name is required")

    session_id = uuid.uuid4().hex[:12]
    db.create_session(session_id, class_name.strip(), source_mode, rtsp_url)
    session_manager.create_engine(session_id)

    if source_mode == "rtsp":
        rtsp_worker.start_worker(session_id, rtsp_url)

    return {"session_id": session_id, "class_name": class_name.strip(), "source_mode": source_mode}


@app.get("/api/sessions")
def list_sessions():
    rows = db.list_sessions()
    return {"sessions": [dict(r) for r in rows]}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    row = db.get_session(session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    result = dict(row)
    worker = rtsp_worker.get_worker(session_id)
    result["camera_error"] = worker.error if worker else None
    return result


@app.post("/api/sessions/{session_id}/stop")
def stop_session(session_id: str):
    row = db.get_session(session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    db.stop_session(session_id)
    rtsp_worker.stop_worker(session_id)
    session_manager.remove_engine(session_id)
    return {"success": True}


# ---------------- Live processing (webcam mode: browser posts frames) ----------------

@app.post("/api/sessions/{session_id}/frame")
async def post_frame(session_id: str, file: UploadFile = File(...)):
    engine = session_manager.get_engine(session_id)
    if engine is None:
        raise HTTPException(404, "Session not active. Was it started or already stopped?")
    data = await file.read()
    frame = session_manager.decode_jpeg_bytes(data)
    if frame is None:
        raise HTTPException(400, "Could not decode frame image")
    result = engine.process_frame(frame)
    # Trim box floats for a lighter payload
    for p in result["people"]:
        p["box"] = [round(v, 1) for v in p["box"]]
    return result


# ---------------- Live processing (RTSP mode: server pulls frames, streams MJPEG) ----------------

def _mjpeg_generator(session_id: str):
    while True:
        worker = rtsp_worker.get_worker(session_id)
        if worker is None:
            break
        jpeg = worker.get_jpeg()
        if jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.1)


@app.get("/api/sessions/{session_id}/stream")
def stream(session_id: str):
    worker = rtsp_worker.get_worker(session_id)
    if worker is None:
        raise HTTPException(404, "No active camera stream for this session")
    return StreamingResponse(_mjpeg_generator(session_id), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------- Live stats (both modes poll this) ----------------

@app.get("/api/sessions/{session_id}/live")
def live_stats(session_id: str):
    engine = session_manager.get_engine(session_id)
    people = engine.last_boxes if engine else []
    headcount = len(people)
    hands_raised = sum(1 for p in people if p["hand_raised"])
    focused_count = sum(1 for p in people if p["focused"] is True)
    distracted_count = sum(1 for p in people if p["focused"] is False)

    alerts = db.get_recent_alerts(session_id, limit=20)
    worker = rtsp_worker.get_worker(session_id)
    attendance_total = db.get_track_count(session_id)

    return {
        "headcount": headcount,
        "hands_raised": hands_raised,
        "focused_count": focused_count,
        "distracted_count": distracted_count,
        "attendance_total_tracked": attendance_total,
        "people": [
            {"track_id": p["track_id"], "display_name": p["display_name"],
             "hand_raised": p["hand_raised"], "head_state": p["head_state"], "focused": p["focused"]}
            for p in people
        ],
        "alerts": [dict(a) for a in alerts],
        "camera_error": worker.error if worker else None,
    }


# ---------------- Tagging (manual, non-biometric roster labeling) ----------------

@app.post("/api/sessions/{session_id}/tag")
def tag_track(session_id: str, track_id: int = Form(...), name: str = Form(...)):
    engine = session_manager.get_engine(session_id)
    if engine is None:
        raise HTTPException(404, "Session not active")
    if not name.strip():
        raise HTTPException(400, "Name cannot be empty")
    ok = engine.tag_track(track_id, name.strip())
    if not ok:
        raise HTTPException(404, "Track not found in this session")
    return {"success": True}


# ---------------- Reports ----------------

def _build_report(session_id: str):
    session_row = db.get_session(session_id)
    if not session_row:
        raise HTTPException(404, "Session not found")

    tracks = db.get_tracks(session_id)
    alerts = db.get_all_alerts(session_id)
    session = dict(session_row)
    total_frames = max(session["frames_processed"], 1)

    attendance = []
    for t in tracks:
        t = dict(t)
        presence_ratio = t["frames_seen"] / total_frames
        focus_ratio = (t["frames_focused"] / t["frames_seen"]) if t["frames_seen"] else None
        attendance.append({
            "track_id": t["track_id"],
            "display_name": t["display_name"] or f"Track #{t['track_id']}",
            "present": presence_ratio >= 0.15,  # seen in at least 15% of processed frames
            "presence_ratio": round(presence_ratio, 3),
            "focus_ratio": round(focus_ratio, 3) if focus_ratio is not None else None,
            "hand_raise_count": t["hand_raise_count"],
            "first_seen": t["first_seen"],
            "last_seen": t["last_seen"],
        })

    present_count = sum(1 for a in attendance if a["present"])

    return {
        "session": session,
        "attendance": attendance,
        "attendance_summary": {
            "total_tracked": len(attendance),
            "present": present_count,
            "absent_or_transient": len(attendance) - present_count,
        },
        "alerts": [dict(a) for a in alerts],
    }


@app.get("/api/sessions/{session_id}/report")
def report(session_id: str):
    return _build_report(session_id)


@app.get("/api/sessions/{session_id}/report.csv")
def report_csv(session_id: str):
    data = _build_report(session_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ClassMonitor Session Report"])
    writer.writerow(["Class", data["session"]["class_name"]])
    writer.writerow(["Started", data["session"]["started_at"]])
    writer.writerow(["Stopped", data["session"]["stopped_at"]])
    writer.writerow([])
    writer.writerow(["Attendance (presence/headcount based, no facial identity)"])
    writer.writerow(["Track", "Name", "Present", "Presence %", "Focus %", "Hand Raises", "First Seen", "Last Seen"])
    for a in data["attendance"]:
        writer.writerow([
            a["track_id"], a["display_name"], "Yes" if a["present"] else "No",
            f"{a['presence_ratio']*100:.0f}%",
            f"{a['focus_ratio']*100:.0f}%" if a["focus_ratio"] is not None else "n/a",
            a["hand_raise_count"], a["first_seen"], a["last_seen"],
        ])
    writer.writerow([])
    writer.writerow(["Alert Log"])
    writer.writerow(["Time", "Type", "Track", "Message"])
    for al in data["alerts"]:
        writer.writerow([al["created_at"], al["type"], al["track_id"], al["message"]])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="classmonitor_{session_id}.csv"'},
    )


# ---------------- Static frontend ----------------

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
