// ---------------- State ----------------
let sessionId = null;
let sourceMode = 'webcam';
let webcamStream = null;
let captureTimer = null;
let liveTimer = null;
let seenAlertIds = new Set();
let pendingTagTrackId = null;
let lastKnownPeople = [];
const CAPTURE_INTERVAL_MS = 600;   // ~1.7 fps sent to the server — plenty for classroom-timescale behavior
const LIVE_POLL_INTERVAL_MS = 1000;
const CAPTURE_MAX_WIDTH = 640;

// ---------------- View switching ----------------
function showView(view) {
  document.getElementById('setupView').classList.toggle('hidden', view !== 'setup');
  document.getElementById('liveView').classList.toggle('hidden', view !== 'live');
  document.getElementById('reportsView').classList.toggle('hidden', view !== 'reports');
  if (view === 'reports') {
    backToSessionsList();
    loadSessionsList();
  }
}

function openModal(id) { document.getElementById(id).classList.remove('hidden'); }
function closeModal(id) { document.getElementById(id).classList.add('hidden'); }

function setSourceMode(mode) {
  sourceMode = mode;
  document.getElementById('modeWebcamBtn').classList.toggle('active', mode === 'webcam');
  document.getElementById('modeRtspBtn').classList.toggle('active', mode === 'rtsp');
  document.getElementById('rtspField').classList.toggle('hidden', mode !== 'rtsp');
}

// ---------------- Starting / stopping a session ----------------

async function startSession() {
  const className = document.getElementById('className').value.trim();
  const msgEl = document.getElementById('setupMsg');
  msgEl.innerHTML = '';

  if (!className) {
    msgEl.innerHTML = '<div class="error-msg">Please enter a class or room name.</div>';
    return;
  }

  const form = new FormData();
  form.append('class_name', className);
  form.append('source_mode', sourceMode);

  if (sourceMode === 'rtsp') {
    const rtspUrl = document.getElementById('rtspUrl').value.trim();
    if (!rtspUrl) {
      msgEl.innerHTML = '<div class="error-msg">Please enter an RTSP stream URL.</div>';
      return;
    }
    form.append('rtsp_url', rtspUrl);
  } else {
    // Ask for webcam access up front so we fail fast with a clear message
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: CAPTURE_MAX_WIDTH } }, audio: false });
    } catch (err) {
      msgEl.innerHTML = `<div class="error-msg">Could not access the webcam: ${escapeHtml(err.message)}</div>`;
      return;
    }
  }

  try {
    const res = await fetch('/api/sessions', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not start session');
    sessionId = data.session_id;
    document.getElementById('liveClassName').textContent = data.class_name;
    seenAlertIds = new Set();
    document.getElementById('alertFeed').innerHTML = '';

    if (sourceMode === 'webcam') {
      const video = document.getElementById('webcamVideo');
      video.srcObject = webcamStream;
      video.classList.remove('hidden');
      document.getElementById('mjpegView').classList.add('hidden');
      video.onloadedmetadata = () => startWebcamCaptureLoop(video);
      document.getElementById('cameraBanner').innerHTML = '';
    } else {
      const img = document.getElementById('mjpegView');
      img.src = `/api/sessions/${sessionId}/stream?t=${Date.now()}`;
      img.classList.remove('hidden');
      document.getElementById('webcamVideo').classList.add('hidden');
      document.getElementById('overlay').getContext('2d').clearRect(0, 0, 9999, 9999);
    }

    showView('live');
    startLivePolling();
  } catch (err) {
    msgEl.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
    if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  }
}

async function stopSession() {
  if (captureTimer) clearInterval(captureTimer);
  if (liveTimer) clearInterval(liveTimer);
  captureTimer = null;
  liveTimer = null;

  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }

  if (sessionId) {
    try { await fetch(`/api/sessions/${sessionId}/stop`, { method: 'POST' }); } catch (e) { /* best effort */ }
  }
  sessionId = null;
  document.getElementById('className').value = '';
  document.getElementById('setupMsg').innerHTML = '';
  showView('setup');
}

// ---------------- Webcam capture loop ----------------

function startWebcamCaptureLoop(video) {
  const captureCanvas = document.createElement('canvas');
  const scale = Math.min(1, CAPTURE_MAX_WIDTH / video.videoWidth);
  captureCanvas.width = Math.round(video.videoWidth * scale);
  captureCanvas.height = Math.round(video.videoHeight * scale);
  const ctx = captureCanvas.getContext('2d');

  const overlay = document.getElementById('overlay');
  overlay.width = captureCanvas.width;
  overlay.height = captureCanvas.height;

  captureTimer = setInterval(async () => {
    if (!sessionId) return;
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCanvas.toBlob(async (blob) => {
      if (!blob || !sessionId) return;
      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      try {
        const res = await fetch(`/api/sessions/${sessionId}/frame`, { method: 'POST', body: form });
        if (!res.ok) return;
        const data = await res.json();
        lastKnownPeople = data.people;
        updateStats(data.headcount, data.hands_raised, data.focused_count, data.distracted_count);
        drawOverlay(data.people, captureCanvas.width, captureCanvas.height);
        renderPeopleList(data.people);
      } catch (err) { /* transient network hiccup, next tick will retry */ }
    }, 'image/jpeg', 0.7);
  }, CAPTURE_INTERVAL_MS);
}

function drawOverlay(people, w, h) {
  const overlay = document.getElementById('overlay');
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  people.forEach((p) => {
    const [x1, y1, x2, y2] = p.box;
    let color = '#a0a0a0';
    if (p.focused === true) color = '#1a9e5a';
    else if (p.focused === false) color = '#d64545';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const name = p.display_name || `#${p.track_id}`;
    const stateText = p.confidence_ok ? headStateLabel(p.head_state) : 'unclear';
    const label = `${name}${p.hand_raised ? ' ✋' : ''} — ${stateText}`;
    ctx.font = '13px sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 4, Math.max(12, y1 - 5));
  });
}

// ---------------- Live polling (stats + alerts; both modes) ----------------

function startLivePolling() {
  pollLive();
  liveTimer = setInterval(pollLive, LIVE_POLL_INTERVAL_MS);
}

async function pollLive() {
  if (!sessionId) return;
  try {
    const res = await fetch(`/api/sessions/${sessionId}/live`);
    if (!res.ok) return;
    const data = await res.json();

    const banner = document.getElementById('cameraBanner');
    if (data.camera_error) {
      banner.innerHTML = `<div class="banner warn">⚠️ ${escapeHtml(data.camera_error)}</div>`;
    } else {
      banner.innerHTML = '';
    }

    if (sourceMode === 'rtsp') {
      // Webcam mode already updates headcount/hands/focus/distracted per-frame from the POST
      // response; RTSP mode relies entirely on this poll since the server renders its own overlay.
      updateStats(data.headcount, data.hands_raised, data.focused_count, data.distracted_count, data.attendance_total_tracked);
      lastKnownPeople = data.people;
      renderPeopleList(data.people);
    } else {
      document.getElementById('statAttendance').textContent = data.attendance_total_tracked;
    }

    renderAlerts(data.alerts);
  } catch (err) { /* transient, next tick retries */ }
}

function updateStats(headcount, hands, focused, distracted, attendanceTotal) {
  document.getElementById('statHeadcount').textContent = headcount;
  document.getElementById('statHands').textContent = hands;
  document.getElementById('statFocused').textContent = focused;
  document.getElementById('statDistracted').textContent = distracted;
  if (attendanceTotal !== undefined) {
    document.getElementById('statAttendance').textContent = attendanceTotal;
  }
}

function renderPeopleList(people) {
  const list = document.getElementById('peopleList');
  const empty = document.getElementById('peopleEmpty');
  if (!people || people.length === 0) {
    list.innerHTML = '';
    empty.classList.remove('hidden');
    return;
  }
  empty.classList.add('hidden');
  list.innerHTML = people.map((p) => {
    const dotClass = p.focused === true ? 'focused' : p.focused === false ? 'distracted' : 'unknown';
    const focusLabel = p.focused === true ? 'Focused' : p.focused === false ? 'Distracted' : 'Unclear';
    const name = escapeHtml(p.display_name || `Track #${p.track_id}`);
    return `
      <div class="person-row">
        <span><span class="state-dot ${dotClass}"></span>${name}
          ${p.hand_raised ? ' ✋' : ''} <span class="note-text">— ${focusLabel} (${headStateLabel(p.head_state)})</span>
        </span>
        <button class="secondary" onclick="openTagModal(${p.track_id})">Label</button>
      </div>`;
  }).join('');
}

function renderAlerts(alerts) {
  const feed = document.getElementById('alertFeed');
  const empty = document.getElementById('alertEmpty');
  if (!alerts || alerts.length === 0) {
    if (feed.children.length === 0) empty.classList.remove('hidden');
    return;
  }
  empty.classList.add('hidden');
  // alerts come newest-first from the API; prepend any we haven't shown yet
  const fresh = alerts.filter(a => !seenAlertIds.has(a.id)).reverse();
  fresh.forEach((a) => {
    seenAlertIds.add(a.id);
    const div = document.createElement('div');
    div.className = `alert-item ${a.type}`;
    div.innerHTML = `${ALERT_LABELS[a.type] || a.type}: ${escapeHtml(a.message)}<div class="alert-time">${formatDate(a.created_at)}</div>`;
    feed.prepend(div);
  });
}

// ---------------- Tagging ----------------

function openTagModal(trackId) {
  pendingTagTrackId = trackId;
  document.getElementById('tagName').value = '';
  document.getElementById('tagMsg').innerHTML = '';
  openModal('tagModal');
}

async function submitTag() {
  const name = document.getElementById('tagName').value.trim();
  const msgEl = document.getElementById('tagMsg');
  if (!name) { msgEl.innerHTML = '<div class="error-msg">Enter a label first.</div>'; return; }
  const form = new FormData();
  form.append('track_id', pendingTagTrackId);
  form.append('name', name);
  try {
    const res = await fetch(`/api/sessions/${sessionId}/tag`, { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not save label');
    closeModal('tagModal');
    renderPeopleList(lastKnownPeople.map(p => p.track_id === pendingTagTrackId ? { ...p, display_name: name } : p));
  } catch (err) {
    msgEl.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
  }
}

// ---------------- Reports ----------------

function backToSessionsList() {
  document.getElementById('reportsListPane').classList.remove('hidden');
  document.getElementById('reportDetailPane').classList.add('hidden');
}

async function loadSessionsList() {
  const el = document.getElementById('sessionsList');
  el.innerHTML = '<div class="spinner-text">Loading...</div>';
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    if (data.sessions.length === 0) {
      el.innerHTML = '<div class="empty-state">No sessions yet. Start one from "New session".</div>';
      return;
    }
    el.innerHTML = data.sessions.map((s) => `
      <div class="session-list-item">
        <span>
          <strong>${escapeHtml(s.class_name)}</strong>
          <span class="note-text"> — ${formatDate(s.started_at)} — ${s.status}</span>
        </span>
        <button class="secondary" onclick="openReport('${s.id}')">View report</button>
      </div>`).join('');
  } catch (err) {
    el.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
  }
}

let currentReportSessionId = null;

async function openReport(id) {
  currentReportSessionId = id;
  document.getElementById('reportsListPane').classList.add('hidden');
  document.getElementById('reportDetailPane').classList.remove('hidden');
  try {
    const res = await fetch(`/api/sessions/${id}/report`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Could not load report');

    document.getElementById('reportClassName').textContent = data.session.class_name;
    document.getElementById('reportMeta').textContent =
      `${formatDate(data.session.started_at)} → ${data.session.stopped_at ? formatDate(data.session.stopped_at) : 'still active'} · ` +
      `${data.attendance_summary.present} present / ${data.attendance_summary.total_tracked} tracked`;

    document.getElementById('attendanceBody').innerHTML = data.attendance.map((a) => `
      <tr>
        <td>${escapeHtml(a.display_name)}</td>
        <td>${a.present ? '<span class="pill positive">Present</span>' : '<span class="pill negative">Not confirmed</span>'}</td>
        <td>${(a.presence_ratio * 100).toFixed(0)}%</td>
        <td>${a.focus_ratio !== null ? (a.focus_ratio * 100).toFixed(0) + '%' : '—'}</td>
        <td>${a.hand_raise_count}</td>
      </tr>`).join('') || '<tr><td colspan="5" class="empty-state">No one was tracked in this session.</td></tr>';

    document.getElementById('reportAlertsBody').innerHTML = data.alerts.map((al) => `
      <tr>
        <td class="note-text">${formatDate(al.created_at)}</td>
        <td>${ALERT_LABELS[al.type] || al.type}</td>
        <td>${al.track_id !== null ? '#' + al.track_id : '—'}</td>
        <td>${escapeHtml(al.message)}</td>
      </tr>`).join('') || '<tr><td colspan="4" class="empty-state">No alerts recorded.</td></tr>';
  } catch (err) {
    document.getElementById('reportMeta').innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
  }
}

function downloadCsv() {
  if (!currentReportSessionId) return;
  window.open(`/api/sessions/${currentReportSessionId}/report.csv`, '_blank');
}

// ---------------- Init ----------------
showView('setup');
