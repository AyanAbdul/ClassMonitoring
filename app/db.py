"""
SQLite storage for ClassMonitor.

Everything the report/attendance features need is written here, so a session's
history survives even if the live in-memory tracker state is gone (e.g. after
a server restart) — the report always reads from the database, never from
memory, which is what makes the numbers reproducible and trustworthy.
"""
import sqlite3
import os
import json
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "classmonitor.db")

_local = threading.local()


def get_conn():
    if not hasattr(_local, "conn"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            class_name TEXT NOT NULL,
            source_mode TEXT NOT NULL CHECK(source_mode IN ('webcam','rtsp')),
            rtsp_url TEXT,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','stopped')),
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            stopped_at TEXT,
            frames_processed INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tracks (
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL,
            display_name TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            frames_seen INTEGER NOT NULL DEFAULT 0,
            frames_focused INTEGER NOT NULL DEFAULT 0,
            frames_distracted INTEGER NOT NULL DEFAULT 0,
            hand_raise_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (session_id, track_id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            track_id INTEGER,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            headcount INTEGER NOT NULL,
            hands_raised INTEGER NOT NULL,
            focused_count INTEGER NOT NULL,
            distracted_count INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id);
        CREATE INDEX IF NOT EXISTS idx_tracks_session ON tracks(session_id);
        """
    )
    conn.commit()


def create_session(session_id, class_name, source_mode, rtsp_url=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (id, class_name, source_mode, rtsp_url) VALUES (?, ?, ?, ?)",
        (session_id, class_name, source_mode, rtsp_url),
    )
    conn.commit()


def stop_session(session_id):
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET status='stopped', stopped_at=datetime('now') WHERE id=?",
        (session_id,),
    )
    conn.commit()


def increment_frames(session_id):
    conn = get_conn()
    conn.execute("UPDATE sessions SET frames_processed = frames_processed + 1 WHERE id=?", (session_id,))
    conn.commit()


def upsert_track(session_id, track_id, focused, hand_raised_event):
    """Update per-track rolling stats for the frame. focused is True/False/None (None = ambiguous, not counted)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM tracks WHERE session_id=? AND track_id=?", (session_id, track_id)
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO tracks (session_id, track_id, first_seen, last_seen, frames_seen,
               frames_focused, frames_distracted, hand_raise_count)
               VALUES (?, ?, datetime('now'), datetime('now'), 1, ?, ?, ?)""",
            (session_id, track_id, 1 if focused else 0, 1 if focused is False else 0, 1 if hand_raised_event else 0),
        )
    else:
        conn.execute(
            """UPDATE tracks SET last_seen=datetime('now'), frames_seen=frames_seen+1,
               frames_focused = frames_focused + ?, frames_distracted = frames_distracted + ?,
               hand_raise_count = hand_raise_count + ?
               WHERE session_id=? AND track_id=?""",
            (
                1 if focused else 0,
                1 if focused is False else 0,
                1 if hand_raised_event else 0,
                session_id,
                track_id,
            ),
        )
    conn.commit()


def tag_track(session_id, track_id, display_name):
    conn = get_conn()
    cur = conn.execute(
        "UPDATE tracks SET display_name=? WHERE session_id=? AND track_id=?",
        (display_name, session_id, track_id),
    )
    conn.commit()
    return cur.rowcount > 0


def add_alert(session_id, track_id, alert_type, message):
    conn = get_conn()
    conn.execute(
        "INSERT INTO alerts (session_id, track_id, type, message) VALUES (?, ?, ?, ?)",
        (session_id, track_id, alert_type, message),
    )
    conn.commit()


def add_snapshot(session_id, headcount, hands_raised, focused_count, distracted_count):
    conn = get_conn()
    conn.execute(
        """INSERT INTO snapshots (session_id, headcount, hands_raised, focused_count, distracted_count)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, headcount, hands_raised, focused_count, distracted_count),
    )
    conn.commit()


def get_session(session_id):
    conn = get_conn()
    return conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()


def list_sessions():
    conn = get_conn()
    return conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()


def get_recent_alerts(session_id, limit=30):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM alerts WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)
    ).fetchall()


def get_all_alerts(session_id):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM alerts WHERE session_id=? ORDER BY id ASC", (session_id,)
    ).fetchall()


def get_track_count(session_id):
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) as c FROM tracks WHERE session_id=?", (session_id,)).fetchone()
    return row["c"] if row else 0


def get_tracks(session_id):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM tracks WHERE session_id=? ORDER BY track_id ASC", (session_id,)
    ).fetchall()


def get_snapshots(session_id):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM snapshots WHERE session_id=? ORDER BY id ASC", (session_id,)
    ).fetchall()
