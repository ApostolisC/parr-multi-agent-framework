/* Shared utility functions. */

export function esc(str) {
  if (str === null || str === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(str);
  return d.innerHTML;
}

export function formatJson(obj) {
  try { return JSON.stringify(obj, null, 2); }
  catch { return String(obj); }
}

export function fmtNum(n) { return Number(n).toLocaleString(); }

export function fmtTime(iso) {
  try { return new Date(iso).toLocaleString(); }
  catch { return iso; }
}

export function fmtDuration(ms) {
  if (!ms) return '0s';
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

export function quickHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return h;
}

export function isFrameworkTool(name) {
  const fwTools = [
    'create_todo_list', 'update_todo_list', 'get_todo_list',
    'mark_todo_complete', 'batch_mark_todo_complete',
    'log_finding', 'batch_log_findings', 'get_findings',
    'review_checklist', 'get_review_summary', 'get_report_template',
    'submit_report', 'spawn_agent', 'wait_for_agents',
    'get_agent_results', 'get_agent_results_all', 'get_agent_result',
    'send_message', 'read_messages', 'set_shared_state', 'get_shared_state',
  ];
  return fwTools.includes(name);
}

/**
 * Classify a tool call for user view rendering.
 * Returns: 'finding' | 'progress' | 'spawn' | 'wait' | 'hidden' | 'domain'
 *  - finding: log_finding / batch_log_findings → render as inline finding
 *  - progress: mark_todo_complete / batch_mark_todo_complete → status line
 *  - spawn: spawn_agent → one-liner showing agent spawn
 *  - wait: wait_for_agents → one-liner + sub-agent dashboard
 *  - hidden: all other framework tools → skip
 *  - domain: user-defined tools → smart summary or collapsed row
 */
export function classifyToolForUserView(name) {
  if (name === 'log_finding' || name === 'batch_log_findings') return 'finding';
  if (name === 'mark_todo_complete' || name === 'batch_mark_todo_complete') return 'progress';
  if (name === 'spawn_agent') return 'spawn';
  if (name === 'wait_for_agents') return 'wait';
  if (isFrameworkTool(name)) return 'hidden';
  return 'domain';
}

export function progressBarRow(label, current, max, pct, colorClass) {
  const maxLabel = max ? ` / ${typeof max === 'number' ? fmtNum(max) : max}` : '';
  const curLabel = typeof current === 'number' ? fmtNum(current) : current;
  return `<div class="progress-row">
    <span class="progress-label">${esc(label)}</span>
    <div class="progress-bar"><div class="progress-fill ${colorClass}" style="width:${pct.toFixed(1)}%"></div></div>
    <span class="progress-value">${curLabel}${maxLabel}</span>
  </div>`;
}

export function pctColor(pct) {
  if (pct >= 90) return 'red';
  if (pct >= 70) return 'yellow';
  return 'green';
}
