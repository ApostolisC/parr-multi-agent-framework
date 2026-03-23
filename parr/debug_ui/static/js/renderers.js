/* Markdown, metrics panel, phases, and tool call renderers. */
import { state } from './state.js';
import { esc, formatJson, fmtNum, fmtDuration, isFrameworkTool, progressBarRow, pctColor } from './utils.js';

// Minimal markdown: headers, bold, code blocks, bullets, paragraphs
export function renderMarkdown(text) {
  if (!text) return '';
  let s = esc(text);
  // Code blocks
  s = s.replace(/```([\s\S]*?)```/g, '<pre style="background:rgba(0,0,0,0.2);padding:8px;border-radius:4px;overflow-x:auto;font-size:11px">$1</pre>');
  // Inline code
  s = s.replace(/`([^`]+)`/g, '<code style="background:rgba(0,0,0,0.2);padding:1px 4px;border-radius:3px;font-size:11px">$1</code>');
  // Headers
  s = s.replace(/^#### (.+)$/gm, '<div style="font-weight:700;font-size:12px;margin:8px 0 4px">$1</div>');
  s = s.replace(/^### (.+)$/gm, '<div style="font-weight:700;font-size:13px;margin:10px 0 4px">$1</div>');
  s = s.replace(/^## (.+)$/gm, '<div style="font-weight:700;font-size:14px;margin:12px 0 6px">$1</div>');
  s = s.replace(/^# (.+)$/gm, '<div style="font-weight:700;font-size:15px;margin:14px 0 6px">$1</div>');
  // Bold
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Bullet lists
  s = s.replace(/^- (.+)$/gm, '<div style="padding-left:16px">&#8226; $1</div>');
  // Newlines to <br> (but not inside pre blocks)
  s = s.replace(/\n/g, '<br>');
  return s;
}

export function renderMetricsPanel(m, budgetLimits) {
  const tokens = m.tokens || {};
  const tools = m.tools || {};
  const phases = m.phases || {};
  const context = m.context || {};
  const activity = m.activity || {};

  let html = `<div class="agent-metrics-grid mb-12">
    <div class="am-card"><div class="am-val">${fmtNum(tokens.input || 0)}</div><div class="am-lbl">Input Tokens</div></div>
    <div class="am-card"><div class="am-val">${fmtNum(tokens.output || 0)}</div><div class="am-lbl">Output Tokens</div></div>
    <div class="am-card"><div class="am-val">${fmtNum(tokens.total || 0)}</div><div class="am-lbl">Total Tokens</div></div>
    <div class="am-card"><div class="am-val">$${(tokens.cost || 0).toFixed(4)}</div><div class="am-lbl">Cost</div></div>
    ${tokens.is_estimated ? `<div class="am-card"><div class="am-val">EST</div><div class="am-lbl">Token Source</div></div>` : ''}
    <div class="am-card"><div class="am-val">${fmtNum(context.estimated_tokens || 0)}</div><div class="am-lbl">Context (Est.)</div></div>
    <div class="am-card"><div class="am-val">${fmtNum(context.chars || 0)}</div><div class="am-lbl">Context Chars</div></div>
    <div class="am-card"><div class="am-val">${tools.total || 0}</div><div class="am-lbl">Tool Calls</div></div>
    <div class="am-card"><div class="am-val">${tools.success || 0}</div><div class="am-lbl">Successful</div></div>
    <div class="am-card"><div class="am-val">${tools.failed || 0}</div><div class="am-lbl">Failed</div></div>
    <div class="am-card"><div class="am-val">${phases.total_iterations || 0}</div><div class="am-lbl">Iterations</div></div>
    ${activity.state ? `<div class="am-card"><div class="am-val">${esc(activity.state)}</div><div class="am-lbl">Activity</div></div>` : ''}
    ${activity.current_phase ? `<div class="am-card"><div class="am-val">${esc(activity.current_phase)}</div><div class="am-lbl">Current Phase</div></div>` : ''}
    ${m.duration_ms ? `<div class="am-card"><div class="am-val">${fmtDuration(m.duration_ms)}</div><div class="am-lbl">Duration</div></div>` : ''}
    ${(m.sub_agents||{}).count ? `<div class="am-card"><div class="am-val">${m.sub_agents.count}</div><div class="am-lbl">Sub-Agents</div></div>` : ''}
  </div>`;

  // Progress bars for budget consumed
  html += '<div class="mb-12">';
  if (budgetLimits.max_tokens) {
    const pct = Math.min(100, ((tokens.total || 0) / budgetLimits.max_tokens) * 100);
    html += progressBarRow('Token Budget', tokens.total || 0, budgetLimits.max_tokens, pct, pctColor(pct));
  }
  if (budgetLimits.max_cost) {
    const pct = Math.min(100, ((tokens.cost || 0) / budgetLimits.max_cost) * 100);
    html += progressBarRow('Cost Budget', `$${(tokens.cost||0).toFixed(4)}`, `$${budgetLimits.max_cost}`, pct, pctColor(pct));
  }
  if (budgetLimits.max_tool_calls) {
    const pct = Math.min(100, ((tools.total || 0) / budgetLimits.max_tool_calls) * 100);
    html += progressBarRow('Tool Budget', tools.total || 0, budgetLimits.max_tool_calls, pct, pctColor(pct));
  }
  if (tools.total) {
    const rate = Math.round((tools.success / tools.total) * 100);
    html += progressBarRow('Success Rate', `${rate}%`, '', rate, rate >= 80 ? 'green' : rate >= 50 ? 'yellow' : 'red');
  }
  html += '</div>';

  // Per-phase breakdown
  if (phases.detail && Object.keys(phases.detail).length) {
    html += `<div class="text-xs text-muted mb-8" style="font-weight:600">Phase Breakdown</div>`;
    for (const [phaseName, pd] of Object.entries(phases.detail)) {
      html += `<div class="flex items-center gap-6 mb-8">
        <span class="tag tag-phase">${esc(phaseName)}</span>
        <span class="text-xs">${pd.iterations} iter</span>
        <span class="text-xs text-muted">${pd.tool_calls} tool calls</span>
        ${pd.hit_limit ? '<span class="tag tag-warn">limit</span>' : ''}
      </div>`;
    }
  }

  // Tool breakdown table
  const byName = tools.by_name || {};
  if (Object.keys(byName).length) {
    html += `<div class="text-xs text-muted mb-8 mt-12" style="font-weight:600">Tool Breakdown</div>
    <table class="tool-breakdown">
      <tr><th>Tool Name</th><th>Calls</th></tr>`;
    const sorted = Object.entries(byName).sort((a, b) => b[1] - a[1]);
    for (const [name, count] of sorted) {
      html += `<tr><td>${esc(name)}</td><td>${count}</td></tr>`;
    }
    html += `</table>`;
  }

  // Iterations per phase (from execution_metadata)
  const ipp = m.iterations_per_phase || {};
  if (Object.keys(ipp).length) {
    html += `<div class="text-xs text-muted mb-8 mt-12" style="font-weight:600">Iterations Per Phase (metadata)</div>
    <div class="kv-grid">`;
    for (const [p, n] of Object.entries(ipp)) {
      html += `<span class="kv-key">${esc(p)}</span><span>${n}</span>`;
    }
    html += `</div>`;
  }

  return html;
}

export function renderPhases(conv, toolCalls, llmCalls, tid) {
  // If we have LLM call data, render per-iteration blocks
  if (llmCalls && llmCalls.length > 0) {
    return _renderPhasesWithIterations(conv, toolCalls, llmCalls, tid);
  }
  // Fall back to flat rendering for old sessions
  return _renderPhasesFlat(conv, toolCalls, tid);
}

function _renderPhasesFlat(conv, toolCalls, tid) {
  let html = '';
  for (const [phase, data] of Object.entries(conv)) {
    const history = data._history || [];
    const revisits = history.length;
    const iters = data.iterations || 0;
    const hitLimit = data.hit_iteration_limit;
    const tcCount = data.tool_calls_count || 0;
    const phaseTools = toolCalls.filter(tc => tc.phase === phase);

    html += `<details class="collapse mb-8" data-cid="ph-${tid}-${phase}">
      <summary>
        <span class="tag tag-phase">${esc(phase)}</span>
        <span class="text-xs text-dim">${iters} iteration${iters !== 1 ? 's' : ''}</span>
        ${tcCount ? `<span class="text-xs text-muted">${tcCount} tool call${tcCount !== 1 ? 's' : ''}</span>` : ''}
        ${hitLimit ? '<span class="tag tag-warn">limit hit</span>' : ''}
        ${revisits ? `<span class="tag tag-warn">visited ${revisits + 1}x</span>` : ''}
      </summary>
      <div class="collapse-body">`;

    // Show previous visit history (collapsed)
    for (let h = 0; h < history.length; h++) {
      const prev = history[h];
      html += `<details class="collapse mb-8 llm-call-block" data-cid="hist-${tid}-${phase}-${h}">
        <summary>
          <span class="text-xs text-dim">Visit ${h + 1}</span>
          <span class="text-xs text-muted">${prev.iterations || 0} iter, ${prev.tool_calls_count || 0} tools</span>
        </summary>
        <div class="collapse-body">`;
      if (prev.content) {
        html += `<div class="json-block">${esc(prev.content)}</div>`;
      }
      html += `</div></details>`;
    }

    if (revisits) {
      html += `<div class="text-xs text-muted mb-4" style="font-weight:600">Latest visit:</div>`;
    }

    if (phaseTools.length) {
      for (let i = 0; i < phaseTools.length; i++) {
        html += renderSingleToolCall(phaseTools[i], `ptc-${tid}-${phase}-${i}`);
      }
    }

    if (data.content) {
      html += `<details class="collapse mt-8" data-cid="llm-${tid}-${phase}">
        <summary><span class="text-xs text-muted">LLM Response</span></summary>
        <div class="collapse-body"><div class="json-block">${esc(data.content)}</div></div>
      </details>`;
    }

    html += `</div></details>`;
  }
  return html;
}

function _renderPhasesWithIterations(conv, toolCalls, llmCalls, tid) {
  let html = '';

  // Group LLM calls by phase
  const llmByPhase = {};
  for (const lc of llmCalls) {
    const p = lc.phase || '_unknown';
    if (!llmByPhase[p]) llmByPhase[p] = [];
    llmByPhase[p].push(lc);
  }

  // Collect all phases: from conv keys + any phase in llmCalls (e.g. "entry")
  const phaseOrder = [];
  const seen = new Set();
  // Entry phase first (if present)
  if (llmByPhase['entry']) {
    phaseOrder.push('entry');
    seen.add('entry');
  }
  for (const p of Object.keys(conv)) {
    if (!seen.has(p)) { phaseOrder.push(p); seen.add(p); }
  }
  for (const p of Object.keys(llmByPhase)) {
    if (!seen.has(p)) { phaseOrder.push(p); seen.add(p); }
  }

  // Index into toolCalls per phase for matching
  const toolsByPhase = {};
  for (const tc of toolCalls) {
    const p = tc.phase || '_unknown';
    if (!toolsByPhase[p]) toolsByPhase[p] = [];
    toolsByPhase[p].push(tc);
  }

  for (const phase of phaseOrder) {
    const data = conv[phase] || {};
    const history = data._history || [];
    const revisits = history.length;
    const phaseLlmCalls = llmByPhase[phase] || [];
    const phaseTools = toolsByPhase[phase] || [];
    // Entry tools may be filed under the detected phase too
    const entryTools = phase !== 'entry' ? (toolsByPhase['entry'] || []) : [];
    const iters = phaseLlmCalls.length || data.iterations || 0;
    const hitLimit = data.hit_iteration_limit;
    const totalToolCount = phaseTools.length + (phase !== 'entry' ? 0 : 0);
    const totalTokensIn = phaseLlmCalls.reduce((s, c) => s + (c.input_tokens || 0), 0);
    const totalTokensOut = phaseLlmCalls.reduce((s, c) => s + (c.output_tokens || 0), 0);
    const hasErrors = phaseLlmCalls.some(c => c.error);
    const hasStalls = phaseLlmCalls.some(c => c.stall_warning);

    html += `<details class="collapse mb-8" data-cid="ph-${tid}-${phase}">
      <summary>
        <span class="tag tag-phase">${esc(phase)}</span>
        <span class="text-xs text-dim">${iters} iteration${iters !== 1 ? 's' : ''}</span>
        <span class="text-xs text-muted">${fmtNum(totalTokensIn)} in / ${fmtNum(totalTokensOut)} out</span>
        ${totalToolCount ? `<span class="text-xs text-muted">${totalToolCount} tool call${totalToolCount !== 1 ? 's' : ''}</span>` : ''}
        ${hitLimit ? '<span class="tag tag-warn">limit hit</span>' : ''}
        ${revisits ? `<span class="tag tag-warn">visited ${revisits + 1}x</span>` : ''}
        ${hasErrors ? '<span class="tag tag-error">errors</span>' : ''}
        ${hasStalls ? '<span class="tag tag-warn">stall</span>' : ''}
      </summary>
      <div class="collapse-body">`;

    // Render each iteration as a block
    let toolIdx = 0;
    for (let i = 0; i < phaseLlmCalls.length; i++) {
      const lc = phaseLlmCalls[i];
      const iterToolCount = (lc.tool_calls || []).length;
      const isError = !!lc.error;
      const isStall = !!lc.stall_warning;
      const blockClass = isError ? ' llm-call-error' : (isStall ? ' llm-call-stall' : '');

      html += `<details class="collapse llm-call-block${blockClass} mb-8" data-cid="iter-${tid}-${phase}-${i}">
        <summary>
          <span class="text-xs text-dim">Iteration ${lc.iteration ?? i}</span>
          <span class="text-xs text-muted">[${fmtNum(lc.input_tokens || 0)} in / ${fmtNum(lc.output_tokens || 0)} out]</span>
          ${iterToolCount ? `<span class="text-xs text-muted">${iterToolCount} tool${iterToolCount !== 1 ? 's' : ''}</span>` : ''}
          ${isStall ? `<span class="tag tag-warn">&#9888; ${esc(lc.stall_warning)}</span>` : ''}
          ${isError ? '<span class="tag tag-error">error</span>' : ''}
        </summary>
        <div class="collapse-body">`;

      // Error content
      if (isError) {
        html += `<div class="llm-error-content">${esc(lc.error)}</div>`;
      }

      // LLM response content (collapsible)
      if (lc.response_content) {
        html += `<details class="collapse mt-4 mb-4" data-cid="resp-${tid}-${phase}-${i}">
          <summary><span class="text-xs text-muted">LLM Response</span></summary>
          <div class="collapse-body"><div class="llm-response-content json-block">${esc(lc.response_content)}</div></div>
        </details>`;
      }

      // Match tool calls from phaseTools by slicing
      if (iterToolCount > 0 && toolIdx < phaseTools.length) {
        const iterTools = phaseTools.slice(toolIdx, toolIdx + iterToolCount);
        // If we got fewer detailed records than expected, show what we have
        const actualCount = Math.min(iterToolCount, iterTools.length);
        for (let t = 0; t < actualCount; t++) {
          html += renderSingleToolCall(iterTools[t], `ptc-${tid}-${phase}-${toolIdx + t}`);
        }
        toolIdx += actualCount;
      }

      html += `</div></details>`;
    }

    // Any remaining tools not matched to iterations (fallback)
    if (toolIdx < phaseTools.length) {
      html += `<div class="text-xs text-muted mt-8 mb-4">Additional tool calls:</div>`;
      for (let t = toolIdx; t < phaseTools.length; t++) {
        html += renderSingleToolCall(phaseTools[t], `ptc-${tid}-${phase}-${t}`);
      }
    }

    // Final LLM response from conversation data (if not already shown)
    if (data.content && !phaseLlmCalls.length) {
      html += `<details class="collapse mt-8" data-cid="llm-${tid}-${phase}">
        <summary><span class="text-xs text-muted">LLM Response</span></summary>
        <div class="collapse-body"><div class="json-block">${esc(data.content)}</div></div>
      </details>`;
    }

    html += `</div></details>`;
  }
  return html;
}

export function renderToolCalls(tools, tid) {
  let html = '';
  for (let i = 0; i < tools.length; i++) {
    html += renderSingleToolCall(tools[i], `tc-${tid}-${i}`);
  }
  return html;
}

export function renderSingleToolCall(tc, cid) {
  const name = tc.name || tc.tool_name || 'unknown';
  const success = tc.success !== false;
  const isFramework = isFrameworkTool(name);
  const tagClass = isFramework ? 'tag-fw' : 'tag-tool';
  const tagText = isFramework ? 'fw' : 'tool';
  const statusTag = success ? '<span class="tag tag-ok">ok</span>' : '<span class="tag tag-fail">fail</span>';

  let html = `<details class="collapse tool-entry mb-8" data-cid="${cid}">
    <summary>
      <span class="tag ${tagClass}">${tagText}</span>
      <span class="tool-name">${esc(name)}</span>
      ${statusTag}
      ${tc.phase ? `<span class="text-xs text-muted">${esc(tc.phase)}</span>` : ''}
    </summary>
    <div class="collapse-body">`;

  // Determine if we have input/output data
  const hasInput = tc.arguments || tc.input;
  const hasResult = tc.result !== undefined && tc.result !== null;
  const hasResultContent = tc.result_content !== undefined && tc.result_content !== null;
  const hasOutput = hasResult || hasResultContent || tc.error;

  if (hasInput || hasOutput) {
    const miniGroup = `mini-${cid}`;
    const defaultTab = hasInput ? 'input' : 'output';
    if (!state.activeTabs[miniGroup]) state.activeTabs[miniGroup] = defaultTab;

    html += `<div class="mini-tabs" data-group="${miniGroup}">`;
    if (hasInput) html += `<button class="mini-tab${state.activeTabs[miniGroup]==='input'?' active':''}" data-tab="input">Input</button>`;
    if (hasOutput) html += `<button class="mini-tab${state.activeTabs[miniGroup]==='output'?' active':''}" data-tab="output">Output</button>`;
    html += `</div>`;

    if (hasInput) {
      const inputData = tc.arguments || tc.input;
      html += `<div class="mini-panel${state.activeTabs[miniGroup]==='input'?' active':''}" data-tab="input">
        <div class="tool-detail">${esc(formatJson(inputData))}</div>
      </div>`;
    }

    if (hasOutput) {
      html += `<div class="mini-panel${state.activeTabs[miniGroup]==='output'?' active':''}" data-tab="output">`;
      if (hasResult) {
        let resultText = typeof tc.result === 'string' ? tc.result : formatJson(tc.result);
        html += `<div class="tool-detail">${esc(resultText)}</div>`;
      }
      if (hasResultContent) {
        html += `<div class="tool-detail">${esc(tc.result_content)}</div>`;
      }
      if (tc.error) {
        html += `<div class="mt-8"><span class="tag tag-error">error</span> <span class="text-sm">${esc(tc.error)}</span></div>`;
      }
      html += `</div>`;
    }
  } else {
    // No data at all - show minimal
    if (tc.error) {
      html += `<div><span class="tag tag-error">error</span> <span class="text-sm">${esc(tc.error)}</span></div>`;
    } else {
      html += `<span class="text-xs text-muted">No detailed input/output recorded.</span>`;
    }
  }

  html += `</div></details>`;
  return html;
}
