/* Global overview panel — token/cost breakdown by agent + tool distribution. */
import { esc, fmtNum } from './utils.js';

/**
 * Render a collapsible global overview with per-agent breakdowns
 * and session-wide tool distribution.
 */
export function renderGlobalOverview(agentTree, globalMetrics) {
  const agents = flattenAgents(agentTree);
  if (agents.length <= 1) return '';

  const totalTokens = (globalMetrics.tokens || {}).total || 1;
  const totalCost = (globalMetrics.tokens || {}).cost || 0;

  let html = `<details class="collapse mb-12" data-cid="global-overview">
    <summary>
      <span class="tag tag-phase">overview</span>
      <strong>Session Breakdown</strong>
      <span class="text-xs text-muted">${agents.length} agents</span>
    </summary>
    <div class="collapse-body">`;

  // -- Token distribution by agent --
  html += '<div class="go-section-title">Token Distribution by Agent</div>';
  html += '<div class="go-bar-chart">';
  const colorCycle = ['', 'green', 'purple', 'orange', 'yellow'];
  for (let i = 0; i < agents.length; i++) {
    const a = agents[i];
    const pct = totalTokens > 0 ? (a.tokens / totalTokens) * 100 : 0;
    const colorClass = colorCycle[i % colorCycle.length];
    const indent = a.depth > 0 ? '\u00A0\u00A0'.repeat(a.depth) : '';
    html += `<div class="go-bar-row">
      <span class="go-bar-label">${indent}${esc(a.label)}</span>
      <div class="go-bar"><div class="go-bar-fill ${colorClass}" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="go-bar-value">${fmtNum(a.tokens)} (${pct.toFixed(0)}%)</span>
    </div>`;
  }
  html += '</div>';

  // -- Cost distribution by agent (only if any cost) --
  if (totalCost > 0) {
    html += '<div class="go-section-title">Cost Distribution by Agent</div>';
    html += '<div class="go-bar-chart">';
    for (let i = 0; i < agents.length; i++) {
      const a = agents[i];
      const pct = totalCost > 0 ? (a.cost / totalCost) * 100 : 0;
      const colorClass = colorCycle[i % colorCycle.length];
      const indent = a.depth > 0 ? '\u00A0\u00A0'.repeat(a.depth) : '';
      html += `<div class="go-bar-row">
        <span class="go-bar-label">${indent}${esc(a.label)}</span>
        <div class="go-bar"><div class="go-bar-fill ${colorClass}" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="go-bar-value">$${a.cost.toFixed(4)} (${pct.toFixed(0)}%)</span>
      </div>`;
    }
    html += '</div>';
  }

  // -- Tool distribution across session --
  const toolTotals = {};
  for (const a of agents) {
    for (const [name, count] of Object.entries(a.toolsByName)) {
      toolTotals[name] = (toolTotals[name] || 0) + count;
    }
  }
  const toolEntries = Object.entries(toolTotals).sort((a, b) => b[1] - a[1]);
  if (toolEntries.length) {
    const maxTools = toolEntries[0][1] || 1;
    html += '<div class="go-section-title">Tool Usage Across Session</div>';
    html += '<div class="go-bar-chart">';
    for (const [name, count] of toolEntries) {
      const pct = (count / maxTools) * 100;
      html += `<div class="go-bar-row">
        <span class="go-bar-label mono">${esc(name)}</span>
        <div class="go-bar"><div class="go-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="go-bar-value">${count}</span>
      </div>`;
    }
    html += '</div>';
  }

  html += '</div></details>';
  return html;
}

/**
 * Flatten agent tree into a list of {label, tokens, cost, tools, toolsByName, depth}.
 */
function flattenAgents(agent, depth = 0) {
  const info = agent.info || {};
  const m = agent.metrics || {};
  const tokens = (m.tokens || {}).total || 0;
  const cost = (m.tokens || {}).cost || 0;
  const tools = (m.tools || {}).total || 0;
  const toolsByName = (m.tools || {}).by_name || {};
  const role = info.role || 'agent';
  const subRole = info.sub_role ? ` / ${info.sub_role}` : '';

  const result = [{
    label: `${role}${subRole}`,
    tokens,
    cost,
    tools,
    toolsByName,
    depth,
  }];

  const subs = agent.sub_agents || {};
  for (const key of Object.keys(subs)) {
    result.push(...flattenAgents(subs[key], depth + 1));
  }
  return result;
}
