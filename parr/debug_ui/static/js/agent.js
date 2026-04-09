/* Recursive agent renderer with tabbed content. */
import { state } from './state.js';
import { esc, fmtNum, fmtDuration, progressBarRow, pctColor } from './utils.js';
import { renderChatView } from './chat.js';
import { renderMetricsPanel, renderPhases, renderToolCalls } from './renderers.js';
import { renderMemory } from './memory.js';
import { renderOutput } from './output.js';
import { renderUserView } from './user-view.js';

export function renderAgent(agent, depth, budgetLimits) {
  const info = agent.info || {};
  const role = info.role || 'agent';
  const subRole = info.sub_role ? ` / ${info.sub_role}` : '';
  const status = info.status || 'unknown';
  const model = info.model || '';
  const task = info.task || '';
  const tid = info.task_id || `d${depth}-${role}`;
  const isUserMode = state.viewMode === 'user';

  // --- User Mode: streamlined card (no tabs, no progress bars) ---
  if (isUserMode) {
    return _renderUserModeAgent(agent, depth, budgetLimits, { role, subRole, status, model, task, tid });
  }

  // --- Debug Mode: full tabbed interface ---
  return _renderDebugModeAgent(agent, depth, budgetLimits, { role, subRole, status, model, task, tid });
}

function _renderUserModeAgent(agent, depth, budgetLimits, ctx) {
  const { role, subRole, status, task, tid } = ctx;

  let html = `<div class="uv-agent-card" data-depth="${depth}">`;

  // Minimal header: no depth labels, no model, no tokens summary
  if (depth === 0) {
    // Root agent gets a clean task display
    if (task) {
      html += `<div class="uv-task">${esc(task)}</div>`;
    }
  }

  // Render user view content (the core stream)
  html += renderUserView(agent, tid);

  html += `</div>`;
  return html;
}

function _renderDebugModeAgent(agent, depth, budgetLimits, ctx) {
  const { role, subRole, status, model, task, tid } = ctx;
  const m = agent.metrics || {};
  const mTokens = m.tokens || {};
  const mTools = m.tools || {};
  const mContext = m.context || {};
  const mActivity = m.activity || {};

  const depthLabel = depth === 0 ? 'Root Agent' : `Sub-Agent (depth ${depth})`;
  const tokenSummary = mTokens.total ? `tokens ${fmtNum(mTokens.total)}` : '';
  const contextSummary = mContext.estimated_tokens ? `ctx ${fmtNum(mContext.estimated_tokens)}` : '';
  const toolSummary = mTools.total ? `${mTools.total} tools` : '';
  const summaryBits = [tokenSummary, contextSummary, toolSummary].filter(Boolean).join(' | ');
  const activityState = mActivity.state || '';
  const activityTag = activityState === 'working'
    ? '<span class="tag tag-blue">working</span>'
    : activityState === 'waiting'
      ? '<span class="tag tag-warn">waiting</span>'
      : '';

  let html = `<details class="collapse agent-card" data-cid="agent-${tid}">
    <summary>
      <span class="tag tag-${depth === 0 ? 'phase' : 'spawn'}">${depthLabel}</span>
      <strong>${esc(role)}${esc(subRole)}</strong>
      <span class="badge badge-${status}">${status}</span>
      ${activityTag}
      ${model ? `<span class="text-xs text-muted">${esc(model)}</span>` : ''}
      ${summaryBits ? `<span class="text-xs text-muted">${summaryBits}</span>` : ''}
    </summary>
    <div class="collapse-body">`;

  if (task) {
    html += `<div class="agent-task">${esc(task)}</div>`;
  }

  // -- Agent-level progress bars --
  html += '<div style="margin-bottom:12px">';
  if (budgetLimits.max_tokens && mTokens.total) {
    const pct = Math.min(100, (mTokens.total / budgetLimits.max_tokens) * 100);
    html += progressBarRow('Tokens', mTokens.total, budgetLimits.max_tokens, pct, pctColor(pct));
  }
  if (budgetLimits.max_cost && mTokens.cost) {
    const pct = Math.min(100, (mTokens.cost / budgetLimits.max_cost) * 100);
    html += progressBarRow('Cost', `$${mTokens.cost.toFixed(4)}`, `$${budgetLimits.max_cost}`, pct, pctColor(pct));
  }
  if (budgetLimits.max_tool_calls && mTools.total) {
    const pct = Math.min(100, (mTools.total / budgetLimits.max_tool_calls) * 100);
    html += progressBarRow('Tools', mTools.total, budgetLimits.max_tool_calls, pct, pctColor(pct));
  }
  if (mTools.total) {
    const rate = Math.round((mTools.success / mTools.total) * 100);
    html += progressBarRow('Success', `${mTools.success}/${mTools.total}`, '', rate, rate >= 80 ? 'green' : rate >= 50 ? 'yellow' : 'red');
  }
  html += '</div>';

  // -- Tabbed content --
  const tabGroup = `agent-tabs-${tid}`;
  const conv = agent.conversation || {};
  const tools = agent.tool_calls || [];
  const mem = agent.memory || {};
  const subs = agent.sub_agents || {};
  const subKeys = Object.keys(subs);

  // Build tab list dynamically
  const tabs = [];
  tabs.push({id: 'chat', label: 'Chat'});
  tabs.push({id: 'overview', label: 'Overview'});
  if (Object.keys(conv).length) tabs.push({id: 'phases', label: `Phases (${Object.keys(conv).length})`});
  if (tools.length) tabs.push({id: 'tools', label: `Tools (${tools.length})`});
  if (Object.keys(mem).length) tabs.push({id: 'memory', label: 'Memory'});
  if (agent.output) tabs.push({id: 'output', label: 'Output'});
  tabs.push({id: 'metrics', label: 'Metrics'});
  if (subKeys.length) tabs.push({id: 'subagents', label: `Sub-Agents (${subKeys.length})`});

  // Default to chat
  if (!state.activeTabs[tabGroup]) state.activeTabs[tabGroup] = 'chat';

  html += `<div class="tabs" data-group="${tabGroup}">`;
  for (const t of tabs) {
    const active = state.activeTabs[tabGroup] === t.id ? ' active' : '';
    html += `<button class="tab-btn${active}" data-tab="${t.id}">${t.label}</button>`;
  }
  html += `</div>`;

  // -- Chat Panel --
  html += `<div class="tab-panel${state.activeTabs[tabGroup]==='chat'?' active':''}" data-tab="chat">`;
  html += renderChatView(agent, tid);
  html += `</div>`;

  // -- Overview Panel --
  html += `<div class="tab-panel${state.activeTabs[tabGroup]==='overview'?' active':''}" data-tab="overview">`;
  html += `<div class="kv-grid mb-12">
    <span class="kv-key">Role</span><span>${esc(role)}${esc(subRole)}</span>
    <span class="kv-key">Model</span><span>${esc(model || 'N/A')}</span>
    <span class="kv-key">Status</span><span>${esc(status)}</span>
    <span class="kv-key">Task ID</span><span class="mono text-xs">${esc(tid)}</span>
    <span class="kv-key">Depth</span><span>${depth}</span>
    ${mTokens.total ? `<span class="kv-key">Tokens</span><span>${fmtNum(mTokens.input)} in / ${fmtNum(mTokens.output)} out = ${fmtNum(mTokens.total)}</span>` : ''}
    ${mTokens.is_estimated ? `<span class="kv-key">Token Source</span><span>Estimated from live data</span>` : ''}
    ${mContext.estimated_tokens ? `<span class="kv-key">Context</span><span>${fmtNum(mContext.estimated_tokens)} est tok (${fmtNum(mContext.chars || 0)} chars)</span>` : ''}
    ${mTokens.cost ? `<span class="kv-key">Cost</span><span>$${mTokens.cost.toFixed(4)}</span>` : ''}
    ${m.duration_ms ? `<span class="kv-key">Duration</span><span>${fmtDuration(m.duration_ms)}</span>` : ''}
    ${mTools.total ? `<span class="kv-key">Tool Calls</span><span>${mTools.success} ok / ${mTools.failed} fail (${mTools.total} total)</span>` : ''}
    ${(m.phases||{}).total_iterations ? `<span class="kv-key">Iterations</span><span>${m.phases.total_iterations}</span>` : ''}
    ${mActivity.current_phase ? `<span class="kv-key">Current Phase</span><span>${esc(mActivity.current_phase)}</span>` : ''}
    ${mActivity.current_doing ? `<span class="kv-key">Current Doing</span><span>${esc(mActivity.current_doing)}</span>` : ''}
    ${subKeys.length ? `<span class="kv-key">Sub-Agents</span><span>${subKeys.length}</span>` : ''}
  </div>`;
  // Show conversation content as system prompt / inputs
  const phaseNames = Object.keys(conv);
  if (phaseNames.length) {
    html += `<div class="text-xs text-muted mb-8">Phases completed: ${phaseNames.map(p => `<span class="tag tag-phase">${esc(p)}</span>`).join(' ')}</div>`;
  }
  html += `</div>`;

  // -- Phases Panel --
  if (Object.keys(conv).length) {
    html += `<div class="tab-panel${state.activeTabs[tabGroup]==='phases'?' active':''}" data-tab="phases">`;
    html += renderPhases(conv, tools, agent.llm_calls || [], tid);
    html += `</div>`;
  }

  // -- Tools Panel --
  if (tools.length) {
    html += `<div class="tab-panel${state.activeTabs[tabGroup]==='tools'?' active':''}" data-tab="tools">`;
    html += renderToolCalls(tools, tid);
    html += `</div>`;
  }

  // -- Memory Panel --
  if (Object.keys(mem).length) {
    html += `<div class="tab-panel${state.activeTabs[tabGroup]==='memory'?' active':''}" data-tab="memory">`;
    html += renderMemory(mem, tid);
    html += `</div>`;
  }

  // -- Output Panel --
  if (agent.output) {
    html += `<div class="tab-panel${state.activeTabs[tabGroup]==='output'?' active':''}" data-tab="output">`;
    html += renderOutput(agent.output, tid);
    html += `</div>`;
  }

  // -- Metrics Panel --
  html += `<div class="tab-panel${state.activeTabs[tabGroup]==='metrics'?' active':''}" data-tab="metrics">`;
  html += renderMetricsPanel(m, budgetLimits);
  html += `</div>`;

  // -- Sub-Agents Panel --
  if (subKeys.length) {
    html += `<div class="tab-panel${state.activeTabs[tabGroup]==='subagents'?' active':''}" data-tab="subagents">`;
    for (const key of subKeys) {
      html += renderAgent(subs[key], depth + 1, budgetLimits);
    }
    html += `</div>`;
  }

  html += `</div></details>`;
  return html;
}
