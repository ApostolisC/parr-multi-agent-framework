/* Start form show/hide/submit logic. */
import { state } from './state.js';
import { api } from './api.js';
import { loadSessions } from './sessions.js';

export function showStartForm() {
  document.getElementById('welcome').style.display = 'none';
  document.getElementById('session-detail').style.display = 'none';
  document.getElementById('start-form').style.display = 'block';
  document.getElementById('sf-task').focus();
}

export function hideStartForm() {
  document.getElementById('start-form').style.display = 'none';
  if (state.currentSessionId) {
    document.getElementById('session-detail').style.display = 'block';
  } else {
    document.getElementById('welcome').style.display = 'flex';
  }
}

export async function submitStartForm() {
  if (!state.config.can_start) return;
  const task = document.getElementById('sf-task').value.trim();
  const role = document.getElementById('sf-role').value;
  if (!task || !role) { alert('Task and role are required.'); return; }

  const btn = document.getElementById('sf-submit');
  btn.disabled = true;
  btn.textContent = 'Starting...';

  try {
    const resp = await api('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, role }),
    });
    if (resp.error) { alert('Error: ' + resp.error); return; }
    document.getElementById('sf-task').value = '';
    hideStartForm();
    state.autoSelectNewest = true;
    await loadSessions();
    if (state.autoSelectNewest) {
      setTimeout(async () => { await loadSessions(); }, 1000);
      setTimeout(async () => { state.autoSelectNewest = false; await loadSessions(); }, 2500);
    }
  } catch (e) {
    alert('Failed to start workflow: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Start Workflow';
  }
}
