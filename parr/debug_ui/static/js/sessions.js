/* Session list and selection logic. */
import { state } from './state.js';
import { api } from './api.js';
import { esc, quickHash } from './utils.js';
import { renderSessionDetail } from './session-detail.js';

export async function loadSessions() {
  const sessions = await api('/api/sessions');
  renderSessionList(sessions);
}

export function renderSessionList(sessions) {
  const container = document.getElementById('session-list');
  const q = state.searchQuery.toLowerCase();
  const filtered = q ? sessions.filter(s => {
    const ag = s.agent_summary || {};
    const searchable = [ag.role, ag.sub_role, ag.task, ag.status, s.workflow_id]
      .filter(Boolean).join(' ').toLowerCase();
    return searchable.includes(q);
  }) : sessions;
  let html;
  if (!filtered.length) {
    html = q
      ? '<div style="padding:20px;color:var(--text-muted);text-align:center;font-size:12px">No matching sessions.</div>'
      : '<div style="padding:20px;color:var(--text-muted);text-align:center;font-size:12px">No sessions found.</div>';
  } else {
    html = '';
    for (const s of filtered) {
      const wf = s.workflow || {};
      const ag = s.agent_summary || {};
      const status = wf.status || ag.status || 'unknown';
      const role = ag.role || 'unknown';
      const subRole = ag.sub_role ? ` / ${ag.sub_role}` : '';
      const active = s.workflow_id === state.currentSessionId ? ' active' : '';
      html += `<div class="session-item${active}" data-wfid="${s.workflow_id}" onclick="selectSession('${s.workflow_id}')">
        <div class="flex items-center gap-6">
          <span class="si-role">${esc(role)}${esc(subRole)}</span>
          <span class="badge badge-${status}">${status}</span>
        </div>
        <div class="si-id">${s.workflow_id.substring(0, 12)}...</div>
        ${ag.task ? `<div class="si-task">${esc(ag.task)}</div>` : ''}
      </div>`;
    }
  }
  const hash = String(quickHash(html));
  if (hash === state.lastSessionListHash) {
    container.querySelectorAll('.session-item').forEach(el => {
      el.classList.toggle('active', el.dataset.wfid === state.currentSessionId);
    });
    return;
  }
  state.lastSessionListHash = hash;
  container.innerHTML = html;
  if (state.autoSelectNewest && sessions.length) {
    state.autoSelectNewest = false;
    const newId = sessions[0].workflow_id;
    // Track conversation chain when continuing from a previous session
    if (state.pendingContinuation) {
      state.chatChains[newId] = state.pendingContinuation;
      state.pendingContinuation = null;
    }
    selectSession(newId);
    // Re-enable chat input now that the new session is selected
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    if (chatInput) { chatInput.disabled = false; }
    if (chatSend) { chatSend.disabled = false; chatSend.innerHTML = '&#9654;'; }
  }
}

export async function selectSession(id) {
  const firstLoad = state.currentSessionId !== id;
  state.currentSessionId = id;
  if (firstLoad) {
    state.openCollapse.clear();
    state.openInlineTools.clear();
    state.openInlineSpawns.clear();
    state.openPhases.clear();
    state.lastDetailHash = '';
  }
  document.querySelectorAll('.session-item').forEach(el => {
    el.classList.toggle('active', el.dataset.wfid === id);
  });
  await loadSessionDetail(id, firstLoad);
  if (state.sessionPollTimer) clearInterval(state.sessionPollTimer);
  const detailPollInterval = state.config.sse_available ? 30000 : 2000;
  state.sessionPollTimer = setInterval(() => {
    if (state.currentSessionId === id) loadSessionDetail(id, false);
  }, detailPollInterval);
}

/** Called by SSE when a session_update event arrives. */
export function onSessionUpdate(eventData) {
  if (state.currentSessionId && eventData.workflow_id === state.currentSessionId) {
    loadSessionDetail(state.currentSessionId, false);
  }
}

export async function cancelSession(workflowId) {
  const resp = await api(`/api/sessions/${workflowId}/cancel`, { method: 'POST' });
  if (!resp.error) {
    loadSessions();
    loadSessionDetail(workflowId, false);
  }
}

export async function loadSessionDetail(id, isFirstLoad) {
  const data = await api(`/api/sessions/${id}`);
  if (!data || data.error) return;
  const dataStr = JSON.stringify(data);
  const hash = String(quickHash(dataStr));
  if (hash === state.lastDetailHash) return;
  state.lastDetailHash = hash;
  renderSessionDetail(data, isFirstLoad);
}
