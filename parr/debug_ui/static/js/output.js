/* Agent output renderer. */
import { esc, formatJson, fmtNum, fmtDuration } from './utils.js';
import { renderMarkdown } from './renderers.js';
import { renderStructuredReport } from './report.js';

export function renderOutput(output, tid) {
  const status = output.status || 'unknown';
  const tokens = output.token_usage || {};
  const meta = output.execution_metadata || {};
  const findings = (output.findings && typeof output.findings === 'object') ? output.findings : null;
  const findingsSummary = findings && findings.summary ? String(findings.summary) : '';

  let html = `<div class="mb-12"><span class="badge badge-${status}">${status}</span></div>`;

  if (output.summary && String(output.summary) !== findingsSummary) {
    html += `<div class="mb-8"><strong class="text-xs text-muted">Summary</strong>
      <div class="agent-task">${esc(output.summary)}</div></div>`;
  }

  if (tokens.input_tokens || tokens.output_tokens) {
    html += `<div class="kv-grid mb-8">
      <span class="kv-key">Input tokens</span><span>${fmtNum(tokens.input_tokens || 0)}</span>
      <span class="kv-key">Output tokens</span><span>${fmtNum(tokens.output_tokens || 0)}</span>
      <span class="kv-key">Total tokens</span><span>${fmtNum(tokens.total_tokens || 0)}</span>
      ${tokens.total_cost ? `<span class="kv-key">Cost</span><span>$${tokens.total_cost.toFixed(4)}</span>` : ''}
    </div>`;
  }

  if (meta.phases_completed && meta.phases_completed.length) {
    html += `<div class="mb-8"><span class="text-xs text-muted">Phases: </span>
      ${meta.phases_completed.map(p => `<span class="tag tag-phase">${esc(p)}</span>`).join(' ')}
    </div>`;
  }

  if (meta.execution_path) {
    const pathLabels = { direct_answer: 'direct answer', full_workflow: 'full workflow', adaptive: 'adaptive' };
    const pathLabel = pathLabels[meta.execution_path] || meta.execution_path;
    html += `<div class="mb-8"><span class="text-xs text-muted">Execution Path: </span>
      <span class="tag tag-phase">${esc(pathLabel)}</span>`;
    if (meta.detected_mode) {
      html += ` <span class="tag tag-status">${esc(meta.detected_mode)}</span>`;
    }
    html += `</div>`;
  }

  if (meta.routing_decision && typeof meta.routing_decision === 'object') {
    const rd = meta.routing_decision;
    const confidence = Number.isFinite(Number(rd.confidence))
      ? Number(rd.confidence).toFixed(2)
      : 'n/a';
    html += `<details class="collapse mt-8" data-cid="out-route-${tid}">
      <summary><span class="text-sm">Routing Decision</span></summary>
      <div class="collapse-body"><div class="kv-grid">
        <span class="kv-key">Selected path</span><span>${esc(String(rd.selected_path || 'n/a'))}</span>
        <span class="kv-key">Mode</span><span>${esc(String(rd.mode || 'n/a'))}</span>
        <span class="kv-key">Confidence</span><span>${esc(confidence)}</span>
        <span class="kv-key">Policy reason</span><span>${esc(String(rd.policy_reason || ''))}</span>
        <span class="kv-key">Reason</span><span>${esc(String(rd.reason || ''))}</span>
      </div></div>
    </details>`;
  }

  if (meta.sub_agents_spawned && meta.sub_agents_spawned.length) {
    html += `<div class="mb-8"><span class="text-xs text-muted">Sub-agents: </span>
      ${meta.sub_agents_spawned.map(s => `<span class="tag tag-spawn">${esc(s.substring(0,8))}</span>`).join(' ')}
    </div>`;
  }

  if (findings && Object.keys(findings).length) {
    html += `<div class="mb-8"><strong class="text-xs text-muted">Submitted Report</strong>`;
    html += renderStructuredReport(findings);
    html += `</div>`;
    html += `<details class="collapse mt-8" data-cid="out-f-${tid}">
      <summary><span class="text-sm">Raw Report JSON</span></summary>
      <div class="collapse-body"><div class="json-block">${esc(formatJson(findings))}</div></div>
    </details>`;
  }

  if (output.errors && output.errors.length) {
    html += `<details class="collapse mt-8" data-cid="out-e-${tid}">
      <summary><span class="tag tag-error">errors</span> Errors (${output.errors.length})</summary>
      <div class="collapse-body">`;
    for (const e of output.errors) {
      html += `<div style="margin-bottom:6px">
        <span class="tag tag-error">${esc(e.error_type || e.source || 'error')}</span>
        <span class="text-sm">${esc(e.message || '')}</span>
      </div>`;
    }
    html += `</div></details>`;
  }

  if (output.recommendations && output.recommendations.length) {
    html += `<details class="collapse mt-8" data-cid="out-r-${tid}">
      <summary><span class="text-sm">Recommendations (${output.recommendations.length})</span></summary>
      <div class="collapse-body"><ul style="padding-left:16px">
        ${output.recommendations.map(r => `<li class="text-sm mb-8">${esc(r)}</li>`).join('')}
      </ul></div>
    </details>`;
  }

  if (meta.phase_outputs && Object.keys(meta.phase_outputs).length) {
    html += `<details class="collapse mt-8" data-cid="out-p-${tid}">
      <summary><span class="text-sm">Phase Outputs</span></summary>
      <div class="collapse-body">`;
    for (const [phase, text] of Object.entries(meta.phase_outputs)) {
      html += `<details class="collapse mb-8" data-cid="out-p-${tid}-${phase}">
        <summary><span class="tag tag-phase">${esc(phase)}</span></summary>
        <div class="collapse-body"><div class="json-block">${esc(text)}</div></div>
      </details>`;
    }
    html += `</div></details>`;
  }

  return html;
}
