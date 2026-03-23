/* Session detail renderer. */
import { state } from './state.js';
import { esc, fmtNum, fmtTime, fmtDuration, progressBarRow, pctColor } from './utils.js';
import { renderAgent } from './agent.js';
import { initTabs } from './tabs.js';
import { saveCollapseState, restoreCollapseState } from './collapse.js';
import { renderAgentTree } from './agent-tree.js';
import { renderGlobalOverview } from './global-overview.js';

export function toggleViewMode() {
  state.viewMode = state.viewMode === 'user' ? 'debug' : 'user';
  if (state.lastSessionData) {
    renderSessionDetail(state.lastSessionData, false);
  }
}

export function renderSessionDetail(data, isFirstLoad) {
  state.lastSessionData = data;
  document.getElementById('welcome').style.display = 'none';
  document.getElementById('start-form').style.display = 'none';
  const el = document.getElementById('session-detail');
  el.style.display = 'block';
  const scrollTop = saveCollapseState();

  const wf = data.workflow || {};
  const status = wf.status || 'unknown';
  const gm = data.global_metrics || {};
  const gmTokens = gm.tokens || {};
  const gmTools = gm.tools || {};
  const gmContext = gm.context || {};
  const gmStates = gm.agent_states || {};
  const limits = gm.budget_limits || {};
  const isUserMode = state.viewMode === 'user';

  const isRunning = status === 'running';
  const canCancel = state.config.can_cancel && isRunning;
  const toggleLabel = isUserMode ? 'Debug' : 'User';
  const toggleIcon = isUserMode ? '&#9881;' : '&#9673;';
  const parentId = state.chatChains[data.workflow_id];

  // Compute duration: live elapsed for running, stored for completed
  let durationDisplay = '';
  if (isRunning && wf.created_at) {
    const elapsed = Date.now() - new Date(wf.created_at).getTime();
    durationDisplay = fmtDuration(elapsed);
  } else if (gm.duration_ms) {
    durationDisplay = fmtDuration(gm.duration_ms);
  }

  let html = `
    <div class="session-header">
      <div class="session-header-top">
        <h2>Session: <span class="mono">${data.workflow_id.substring(0, 16)}...</span></h2>
        <button class="btn btn-sm btn-view-toggle" onclick="toggleViewMode()" title="Switch to ${toggleLabel} view">${toggleIcon} ${toggleLabel} view</button>
      </div>
      ${parentId ? `<div class="chat-chain-link">Continued from <a href="#" onclick="selectSession('${parentId}');return false" class="mono">${parentId.substring(0, 12)}...</a></div>` : ''}
      <div class="session-meta">
        <div class="meta-item"><span class="badge badge-${status}">${status}</span></div>
        ${wf.created_at ? `<div class="meta-item"><span class="meta-label">Created:</span> ${fmtTime(wf.created_at)}</div>` : ''}
        ${gm.agent_count ? `<div class="meta-item"><span class="meta-label">Agents:</span> ${gm.agent_count}</div>` : ''}
        ${durationDisplay && isUserMode ? `<div class="meta-item"><span class="meta-label">Duration:</span> <span id="live-duration">${durationDisplay}</span></div>` : ''}
        ${canCancel ? `<div class="meta-item"><button class="btn btn-cancel" onclick="cancelSession('${data.workflow_id}')">Cancel</button></div>` : ''}
      </div>
    </div>`;

  // -- Debug-only sections: Metrics Dashboard, Progress Bars, Overview, Agent Tree --
  if (!isUserMode) {
    html += `<div class="metrics-dashboard">
      <div class="metric-card">
        <div class="mc-label">Total Tokens</div>
        <div class="mc-value">${fmtNum(gmTokens.total || 0)}</div>
        <div class="mc-sub">${fmtNum(gmTokens.input || 0)} in / ${fmtNum(gmTokens.output || 0)} out</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Total Cost</div>
        <div class="mc-value">$${(gmTokens.cost || 0).toFixed(4)}</div>
        ${limits.max_cost ? `<div class="mc-sub">of $${limits.max_cost} budget</div>` : ''}
      </div>
      <div class="metric-card">
        <div class="mc-label">Tool Calls</div>
        <div class="mc-value">${gmTools.total || 0}</div>
        <div class="mc-sub">${gmTools.success || 0} ok / ${gmTools.failed || 0} failed</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Iterations</div>
        <div class="mc-value">${gm.total_iterations || 0}</div>
        <div class="mc-sub">${gm.agent_count || 1} agent${(gm.agent_count||1) > 1 ? 's' : ''}</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Agent Activity</div>
        <div class="mc-value">${gmStates.working || 0} working</div>
        <div class="mc-sub">${gmStates.waiting || 0} waiting</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">Context (Est.)</div>
        <div class="mc-value">${fmtNum(gmContext.estimated_tokens || 0)}</div>
        <div class="mc-sub">${fmtNum(gmContext.chars || 0)} chars</div>
      </div>
      ${gm.duration_ms ? `<div class="metric-card">
        <div class="mc-label">Duration</div>
        <div class="mc-value">${fmtDuration(gm.duration_ms)}</div>
      </div>` : ''}
    </div>`;

    // -- Global Progress Bars --
    html += '<div style="margin-bottom:20px">';
    if (limits.max_tokens) {
      const pct = Math.min(100, ((gmTokens.total || 0) / limits.max_tokens) * 100);
      html += progressBarRow('Token Budget', gmTokens.total || 0, limits.max_tokens, pct, pctColor(pct));
    }
    if (limits.max_cost) {
      const pct = Math.min(100, ((gmTokens.cost || 0) / limits.max_cost) * 100);
      html += progressBarRow('Cost Budget', `$${(gmTokens.cost||0).toFixed(4)}`, `$${limits.max_cost}`, pct, pctColor(pct));
    }
    if (limits.max_tool_calls) {
      const pct = Math.min(100, ((gmTools.total || 0) / limits.max_tool_calls) * 100);
      html += progressBarRow('Tool Budget', gmTools.total || 0, limits.max_tool_calls, pct, pctColor(pct));
    }
    if (gmTools.total) {
      const rate = gmTools.total ? Math.round((gmTools.success / gmTools.total) * 100) : 100;
      html += progressBarRow('Tool Success', `${gmTools.success}/${gmTools.total}`, '', rate, rate >= 80 ? 'green' : rate >= 50 ? 'yellow' : 'red');
    }
    html += '</div>';

    // -- Global Overview (per-agent breakdown + tool distribution) --
    if (data.agent_tree) {
      html += renderGlobalOverview(data.agent_tree, gm);
    }

    // -- Agent Tree Visualization --
    if (data.agent_tree) {
      html += renderAgentTree(data.agent_tree);
    }
  }

  // Render root agent tree
  if (data.agent_tree) {
    // Inject workflow-level status so Chat view can show failure reasons
    // even when agent.json status is stale (e.g. still "running")
    if (wf.status && data.agent_tree.info) {
      data.agent_tree._workflow_status = wf.status;
    }
    html += renderAgent(data.agent_tree, 0, gm.budget_limits || {});
  }

  el.innerHTML = html;

  // Live duration timer for running sessions
  if (state._durationTimer) { clearInterval(state._durationTimer); state._durationTimer = null; }
  if (isRunning && wf.created_at && isUserMode) {
    const createdTs = new Date(wf.created_at).getTime();
    state._durationTimer = setInterval(() => {
      const el = document.getElementById('live-duration');
      if (el) {
        el.textContent = fmtDuration(Date.now() - createdTs);
      } else {
        clearInterval(state._durationTimer);
        state._durationTimer = null;
      }
    }, 1000);
  }

  // Initialize all tabs
  initTabs();

  if (isFirstLoad) {
    const root = el.querySelector('details[data-cid]');
    if (root) { root.open = true; state.openCollapse.add(root.dataset.cid); }
  } else {
    restoreCollapseState(scrollTop);
  }

  // Show chat input bar when session is completed and continue is available
  const chatBar = document.getElementById('chat-input-bar');
  if (chatBar) {
    const showChat = state.config.can_continue
      && (status === 'completed' || status === 'failed');
    chatBar.classList.toggle('hidden', !showChat);
  }
}
