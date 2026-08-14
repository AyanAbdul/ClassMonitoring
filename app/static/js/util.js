function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso.includes('Z') ? iso : iso.replace(' ', 'T') + 'Z');
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' });
}

function headStateLabel(state) {
  return {
    forward: 'Facing forward',
    down: 'Looking down',
    turned_left: 'Turned away (left)',
    turned_right: 'Turned away (right)',
    unknown: 'Unclear',
  }[state] || state;
}

const ALERT_LABELS = {
  hand_raised: '✋ Hand raised',
  prolonged_distraction: '⚠️ Prolonged distraction',
  student_entered: '🟢 New student detected',
};
