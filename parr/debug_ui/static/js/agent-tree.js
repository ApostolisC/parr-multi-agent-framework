/* Agent tree visualization — CSS-based hierarchy diagram. */
import { esc, fmtNum } from './utils.js';

/**
 * Render a visual tree of the agent hierarchy.
 * Only shown when the root agent has sub-agents.
 */
export function renderAgentTree(agent) {
  const subs = agent.sub_agents || {};
  if (!Object.keys(subs).length) return '';

  let html = '<div class="agent-tree">';
  html += '<ul>';
  html += renderTreeNode(agent, true);
  html += '</ul>';
  html += '</div>';
  return html;
}

function renderTreeNode(agent, isRoot) {
  const info = agent.info || {};
  const m = agent.metrics || {};
  const tokens = m.tokens || {};
  const role = info.role || 'agent';
  const subRole = info.sub_role ? ` / ${info.sub_role}` : '';
  const status = info.status || 'unknown';
  const tid = info.task_id || role;

  let html = '<li>';
  html += `<div class="at-node" onclick="navigateToAgent('${esc(tid)}')">`;
  html += `<span class="at-role">${esc(role)}${esc(subRole)}</span>`;
  html += `<span class="badge badge-${status}">${status}</span>`;
  if (tokens.total) {
    html += `<span class="at-tokens">${fmtNum(tokens.total)} tok</span>`;
  }
  if (tokens.cost) {
    html += `<span class="at-cost">$${tokens.cost.toFixed(4)}</span>`;
  }
  html += '</div>';

  const subs = agent.sub_agents || {};
  const subKeys = Object.keys(subs);
  if (subKeys.length) {
    html += '<ul>';
    for (const key of subKeys) {
      html += renderTreeNode(subs[key], false);
    }
    html += '</ul>';
  }

  html += '</li>';
  return html;
}
