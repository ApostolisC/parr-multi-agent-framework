/* Sub-agent dashboard — live progress bars during wait_for_agents. */
import { esc } from './utils.js';

/**
 * Render a compact dashboard showing all sub-agent progress.
 * Each agent gets a 4-segment progress bar (PLAN | ACT | REVIEW | REPORT).
 * Click to expand full sub-agent event stream.
 *
 * @param {Object} subAgents  - sub_agents dict from agent data
 * @param {string} tid        - parent task ID for unique element IDs
 * @param {number} depth      - nesting depth
 * @param {Array}  parentErrors - parent error list (for budget context)
 * @param {Function} renderStreamFn - recursive renderer (agent, tid, depth) => html
 */
export function renderSubAgentDashboard(subAgents, tid, depth, parentErrors, renderStreamFn) {
  const entries = Object.entries(subAgents);
  if (!entries.length) return '';

  const doneCount = entries.filter(([_, sa]) => _saStatus(sa) === 'completed').length;
  const failCount = entries.filter(([_, sa]) => _saStatus(sa) === 'failed').length;

  let html = '<div class="uv-sa-dashboard">';
  html += '<div class="uv-sa-dash-header">';
  html += `<span class="uv-sa-dash-title">Sub-Agent Progress</span>`;
  html += `<span class="uv-sa-dash-count">${doneCount}/${entries.length} done${failCount ? `, ${failCount} failed` : ''}</span>`;
  html += '</div>';

  // Detect parent failure type for orphaned sub-agents
  let limitDetail = 'budget';
  if (parentErrors && parentErrors.length) {
    for (const err of parentErrors) {
      const msg = err.message || err.error || '';
      const m = msg.match(/^\[(tokens|cost|duration)\]/);
      if (m) { limitDetail = m[1]; break; }
    }
  }

  for (const [saId, sa] of entries) {
    const saInfo   = sa.info || {};
    const saRole   = saInfo.role || saId;
    const saTask   = saInfo.task || '';
    const saStatus = _saStatus(sa);
    const saMetrics = sa.metrics || {};
    const phasesCompleted = (saMetrics.phases?.completed) || [];
    const currentPhase    = saMetrics.activity?.current_phase || '';
    const toolCount       = saMetrics.tools?.total || 0;
    const isLive = saStatus === 'running' || saStatus === 'spawned' || saStatus === 'queued';

    // Annotate orphaned sub-agents (killed by parent budget)
    if (saStatus === 'failed' && !(sa.output || {}).errors) {
      sa._parent_failure_context = `Terminated - parent workflow ${limitDetail} limit exceeded`;
    }

    // Status icon
    const statusIcon = saStatus === 'completed' ? '\u2713'
      : saStatus === 'failed' ? '\u2717'
      : isLive ? '\u23F3'
      : '\u2014';
    const statusCls = saStatus === 'completed' ? 'uv-sa-st-done'
      : saStatus === 'failed' ? 'uv-sa-st-fail'
      : isLive ? 'uv-sa-st-active'
      : '';

    html += `<div class="cs-spawn" data-cssid="uv-sa-${tid}-${saId}">`;
    html += `<div class="uv-sa-dash-card" onclick="toggleSpawnRow(this)">`;

    // Role name
    html += `<span class="uv-sa-dash-role">${esc(saRole)}</span>`;

    // Detect execution mode for progress bar rendering
    const detectedMode = ((sa.output || {}).execution_metadata || {}).detected_mode || '';
    const isDirectAnswer = detectedMode === 'direct_answer';
    const iterPerPhase = ((sa.output || {}).execution_metadata || {}).iterations_per_phase || {};

    // Progress bar: 1 box for direct answer, 4 for normal
    html += '<div class="uv-sa-progress">';
    if (isDirectAnswer) {
      // Single box for direct answer agents
      const segCls = saStatus === 'completed' ? 'uv-sa-seg-done'
        : saStatus === 'failed' ? 'uv-sa-seg-fail'
        : isLive ? 'uv-sa-seg-active'
        : 'uv-sa-seg-future';
      html += `<div class="uv-sa-seg uv-sa-seg-single ${segCls}" title="Direct Answer"></div>`;
    } else {
      const allPhases = ['plan', 'act', 'review', 'report'];
      for (const p of allPhases) {
        const visitCount = phasesCompleted.filter(v => v === p).length;
        const isDone    = visitCount > 0;
        const isRerun   = visitCount > 1;
        const isCurrent = currentPhase === p && isLive;
        const segCls = isDone
          ? (isRerun ? 'uv-sa-seg-rerun' : 'uv-sa-seg-done')
          : isCurrent ? 'uv-sa-seg-active'
          : 'uv-sa-seg-future';
        const title = isRerun ? `${p.toUpperCase()} (x${visitCount})` : p.toUpperCase();
        html += `<div class="uv-sa-seg ${segCls}" title="${title}"></div>`;
      }
    }
    html += '</div>';

    // Phase label
    const phaseLabel = isDirectAnswer
      ? (saStatus === 'completed' ? 'DIRECT' : saStatus === 'failed' ? 'FAIL' : 'THINKING')
      : isLive && currentPhase ? currentPhase.toUpperCase()
      : saStatus === 'completed' ? 'DONE'
      : saStatus === 'failed' ? 'FAIL'
      : '';
    html += `<span class="uv-sa-dash-phase">${esc(phaseLabel)}</span>`;

    // Tool count
    html += `<span class="uv-sa-dash-tools">${toolCount} tool${toolCount !== 1 ? 's' : ''}</span>`;

    // Status
    html += `<span class="uv-sa-dash-status ${statusCls}">${statusIcon}</span>`;

    html += '</div>'; // card

    // Expandable detail (full sub-agent stream)
    html += '<div class="cs-spawn-detail">';
    if (renderStreamFn) {
      html += renderStreamFn(sa, `${tid}-sa-${saId}`, depth + 1);
    }
    html += '</div>';

    html += '</div>'; // cs-spawn
  }

  html += '</div>';
  return html;
}

function _saStatus(sa) {
  const info = sa.info || {};
  const output = sa.output || {};
  return sa._workflow_status || info.status || output.status || 'unknown';
}
