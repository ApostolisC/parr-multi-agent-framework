/* Entry point: init, event wiring, window bridges. */
import { state } from './state.js';
import { api } from './api.js';
import { loadSessions, selectSession, onSessionUpdate, cancelSession } from './sessions.js';
import { showStartForm, hideStartForm, submitStartForm } from './start-form.js';
import { toggleToolRow, toggleSpawnRow, togglePhaseSection, toggleBlock } from './collapse.js';
import { connectSSE } from './sse.js';
import { toggleViewMode } from './session-detail.js';

// Expose functions used by inline onclick handlers in generated HTML
window.selectSession = selectSession;
window.cancelSession = cancelSession;
window.toggleToolRow = toggleToolRow;
window.toggleSpawnRow = toggleSpawnRow;
window.togglePhaseSection = togglePhaseSection;
window.toggleBlock = toggleBlock;
window.togglePill = togglePill;
window.toggleViewMode = toggleViewMode;
window.navigateToAgent = navigateToAgent;
window.retryFailedSession = retryFailedSession;

async function retryFailedSession(taskId) {
  const sessionId = taskId || state.currentSessionId;
  if (!sessionId) return;
  try {
    const resp = await api(`/api/sessions/${sessionId}/continue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'Please retry the previous task. The last attempt failed due to an error.' }),
    });
    if (resp.error) {
      alert(`Retry failed: ${resp.error}`);
      return;
    }
    state.pendingContinuation = sessionId;
    state.autoSelectNewest = true;
    loadSessions();
  } catch (e) {
    alert(`Retry error: ${e.message}`);
  }
}

function navigateToAgent(agentId) {
  const details = document.querySelector(`details[data-cid="agent-${agentId}"]`);
  if (details) {
    details.open = true;
    details.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function togglePill(btn) {
  if (!btn) return;
  const panel = btn.nextElementSibling;
  if (!panel) return;
  const isOpen = panel.style.display === 'block';
  panel.style.display = isOpen ? 'none' : 'block';
  btn.classList.toggle('active', !isOpen);
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const message = (input.value || '').trim();
  if (!message || !state.currentSessionId) return;

  // Disable while sending — stays disabled until new session is auto-selected
  input.disabled = true;
  sendBtn.disabled = true;
  sendBtn.textContent = '...';

  try {
    const resp = await api(`/api/sessions/${state.currentSessionId}/continue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    if (resp.error) {
      alert(`Failed to continue: ${resp.error}`);
      input.disabled = false;
      sendBtn.disabled = false;
      sendBtn.innerHTML = '&#9654;';
      return;
    }
    // Track the chain and auto-select the new session.
    // Input stays disabled until autoSelectNewest resolves in renderSessionList(),
    // preventing double-sends that branch from the wrong session.
    state.pendingContinuation = state.currentSessionId;
    state.autoSelectNewest = true;
    input.value = '';
    loadSessions();
  } catch (e) {
    alert(`Error: ${e.message}`);
    input.disabled = false;
    sendBtn.disabled = false;
    sendBtn.innerHTML = '&#9654;';
  }
}

async function init() {
  state.config = await api('/api/config');
  if (!state.config.can_start) {
    document.getElementById('sf-disabled-msg').classList.remove('hidden');
    document.getElementById('sf-submit').disabled = true;
    document.getElementById('sf-submit').style.opacity = '0.4';
  }
  // Populate role dropdown
  const roleSelect = document.getElementById('sf-role');
  if (state.config.available_roles && state.config.available_roles.length) {
    state.config.available_roles.forEach(r => {
      const o = document.createElement('option');
      o.value = r;
      o.textContent = r;
      roleSelect.appendChild(o);
    });
    if (state.config.available_roles.length === 1) roleSelect.selectedIndex = 1;
  } else {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = 'No roles available';
    o.disabled = true;
    roleSelect.appendChild(o);
  }
  loadSessions();
  const listPollInterval = state.config.sse_available ? 30000 : 3000;
  state.pollTimer = setInterval(loadSessions, listPollInterval);

  // Connect SSE for real-time updates when available
  if (state.config.sse_available) {
    connectSSE({
      onSessionList: () => loadSessions(),
      onSessionUpdate: (data) => onSessionUpdate(data),
    });
  }

  document.getElementById('new-btn').addEventListener('click', showStartForm);
  document.getElementById('sf-submit').addEventListener('click', submitStartForm);
  document.getElementById('sf-cancel').addEventListener('click', hideStartForm);
  document.getElementById('session-search').addEventListener('input', (e) => {
    state.searchQuery = e.target.value;
    loadSessions();
  });

  // Chat follow-up input
  document.getElementById('chat-send').addEventListener('click', sendChatMessage);
  document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
  // Auto-resize textarea
  document.getElementById('chat-input').addEventListener('input', (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  });
}

init();
