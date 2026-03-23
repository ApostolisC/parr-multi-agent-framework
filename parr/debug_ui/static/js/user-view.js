/* User view v3 — Chronological event stream with > one-liners. */
import { esc, classifyToolForUserView } from './utils.js';
import { renderMarkdown } from './renderers.js';
import { _buildLiveTodoList } from './chat.js';
import { renderUserReport } from './user-report.js';
import { renderSubAgentDashboard } from './sub-agent-dashboard.js';
import { state } from './state.js';

// ---------------------------------------------------------------------------
// Tool summary templates — map known domain tools to user-friendly one-liners.
// ---------------------------------------------------------------------------
const TOOL_SUMMARIES = {
  search_documents: (a) => ({ text: `Searching: "${a?.query || a?.search_query || 'documents'}"` }),
  read_section:     (a) => ({ text: `Reading: ${a?.title || a?.section || a?.section_id || a?.name || 'section'}` }),
  read_document:    (a) => ({ text: `Reading: ${a?.title || a?.document || a?.document_id || a?.name || 'document'}` }),
  write_file:       (a) => ({ text: `Writing: ${a?.path || a?.filename || 'file'}` }),
  run_command:      (a) => ({ text: `Running: ${_truncate(a?.command || 'command', 60)}` }),
  execute_code:     (a) => ({ text: `Executing: ${a?.language || 'code'}` }),
  web_search:       (a) => ({ text: `Web search: "${a?.query || 'query'}"` }),
  fetch_url:        (a) => ({ text: `Fetching: ${_truncate(a?.url || 'URL', 50)}` }),
};

const PHASE_LABELS = {
  plan: 'Planning...',
  act: 'Working...',
  review: 'Reviewing...',
  report: 'Preparing report...',
};

/**
 * Render a clean, user-facing view of an agent's work.
 * Every intermediate event is a uniform `>` one-liner that expands on click.
 * Only the final answer gets visual prominence.
 */
export function renderUserView(agent, tid) {
  return _renderUserStream(agent, tid, 0);
}

function _resolveStatus(agent) {
  const info = agent.info || {};
  const output = agent.output || {};
  return agent._workflow_status || info.status || output.status || 'unknown';
}

function _isRunning(status) {
  return status === 'running' || status === 'spawned' || status === 'queued';
}

// ---------------------------------------------------------------------------
// Core recursive renderer
// ---------------------------------------------------------------------------
function _renderUserStream(agent, tid, depth) {
  const conv   = agent.conversation || {};
  const tools  = agent.tool_calls || [];
  const llmCalls = agent.llm_calls || [];
  const subAg  = agent.sub_agents || {};
  const info   = agent.info || {};
  const output = agent.output || {};
  const status = _resolveStatus(agent);
  const metrics  = agent.metrics || {};
  const activity = metrics.activity || {};
  const phaseOut = (output.execution_metadata || {}).phase_outputs || {};
  const phaseOrd = ['plan', 'act', 'review', 'report'];
  const phases   = phaseOrd.filter(p => conv[p] || phaseOut[p]);
  const isLive   = _isRunning(status);
  const hasSub   = Object.keys(subAg).length > 0;

  // Determine final report
  const finalReport = _hasRenderableValue(output.findings)
    ? output.findings
    : (_hasRenderableValue(phaseOut.report) ? phaseOut.report : phaseOut.act);
  const hasFinal = _hasRenderableValue(finalReport);

  // Build todo list
  const todoItems = _buildLiveTodoList(agent);
  const todoDone  = todoItems.filter(t => t.completed).length;

  let html = '<div class="uv-stream">';

  // --- Header ---
  html += `<div class="uv-header">`;
  html += `<span class="uv-role">${esc(info.role || 'Agent')}${esc(info.sub_role ? ` / ${info.sub_role}` : '')}</span>`;
  html += _renderStatusIndicator(status, activity);
  html += `</div>`;

  // --- Plan checklist (always visible) ---
  if (todoItems.length) {
    html += _renderPlanChecklist(todoItems, todoDone, isLive);
  }

  // --- Empty state ---
  if (!phases.length && !tools.length && status !== 'failed' && !hasFinal) {
    html += isLive
      ? `<div class="uv-thinking-anim">Thinking...</div>`
      : '<div class="uv-empty">No content yet.</div>';
    html += '</div>';
    return html;
  }

  // --- Sort tools by sequence number if available ---
  const sortedTools = [...tools];
  if (sortedTools.length && sortedTools.some(t => typeof t.seq === 'number')) {
    sortedTools.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
  }

  // --- Chronological event stream ---
  const events = _buildEventStream(conv, sortedTools, phases, llmCalls);
  let dashboardRendered = false;

  // Pre-scan for last spawn index (to render dashboard after last spawn)
  const lastSpawnIdx = events.reduce((acc, evt, i) => evt.type === 'spawn' ? i : acc, -1);

  for (let i = 0; i < events.length; i++) {
    const evt = events[i];
    switch (evt.type) {
      case 'phase_start':
        html += _renderPhaseMarker(evt.phase, isLive && evt.isLast);
        break;
      case 'finding':
        if (!hasSub) html += _renderFindingEvent(evt.tc, `${tid}-${evt.id}`);
        break;
      case 'progress':
        html += _renderProgressEvent(evt.tc, todoItems);
        break;
      case 'spawn':
        html += _renderSpawnEvent(evt.tc, `${tid}-${evt.id}`);
        // Render dashboard after last spawn (before wait)
        if (hasSub && i === lastSpawnIdx) {
          html += renderSubAgentDashboard(subAg, tid, depth, output.errors, _renderUserStream);
          dashboardRendered = true;
        }
        break;
      case 'wait':
        html += _renderWaitEvent(evt.tc, `${tid}-${evt.id}`, subAg);
        break;
      case 'domain':
        html += _renderDomainEvent(evt.tc, `${tid}-${evt.id}`);
        break;
      case 'thinking':
        html += _renderThinkingEvent(evt.content, evt.phase, tid);
        break;
      case 'reasoning':
        html += _renderReasoningLine(evt.content);
        break;
      case 'review':
        html += _renderReviewBadge(evt.passed);
        break;
      case 'content':
        html += `<div class="uv-content">${renderMarkdown(evt.content)}</div>`;
        break;
    }
  }

  // Sub-agents fallback (if no spawn events were emitted yet)
  if (hasSub && !dashboardRendered) {
    html += renderSubAgentDashboard(subAg, tid, depth, output.errors, _renderUserStream);
  }

  // --- Live working spinner at bottom ---
  if (isLive) {
    const doing = activity.current_doing || activity.current_phase || 'Working';
    html += `<div class="uv-live-spinner"><span class="uv-pulse"></span><span>${esc(doing)}...</span></div>`;
  }

  // --- Error banner ---
  if (status === 'failed' || status === 'error') {
    html += _renderErrorBanner(agent);
  }

  // --- Final report (prominent, with separator) ---
  if (hasFinal) {
    html += '<div class="uv-answer-separator"></div>';
    html += '<div class="uv-answer-section">';
    if (typeof finalReport === 'object' && finalReport !== null) {
      html += renderUserReport(finalReport);
    } else {
      html += `<div class="uv-answer-prose">${renderMarkdown(
        typeof finalReport === 'string' ? finalReport : JSON.stringify(finalReport, null, 2)
      )}</div>`;
    }
    html += '</div>';
  }

  html += '</div>';
  return html;
}

// ---------------------------------------------------------------------------
// Event stream builder — chronological list from tool_calls + conv phases
// ---------------------------------------------------------------------------
function _buildEventStream(conv, tools, phases, llmCalls) {
  const events = [];
  const stdPhases = ['plan', 'act', 'review', 'report'];

  // Build a map from llm_calls: iteration → response_content (for reasoning lines)
  const llmReasoningByPhase = {};
  if (llmCalls && llmCalls.length) {
    for (const lc of llmCalls) {
      const p = lc.phase === 'entry'
        ? (phases.includes('plan') ? 'plan' : 'act')
        : (lc.phase || '');
      if (!llmReasoningByPhase[p]) llmReasoningByPhase[p] = [];
      if (lc.response_content && lc.response_content.trim()) {
        llmReasoningByPhase[p].push(lc.response_content.trim());
      }
    }
  }

  // Assign phase to tools that lack the field (legacy sessions)
  const phaseMap = new Map();
  const hasPhaseField = tools.some(t => t.phase);
  if (!hasPhaseField && phases.length) {
    let idx = 0;
    for (const p of phases) {
      const count = (conv[p] || {}).tool_calls_count || 0;
      for (let k = 0; k < count && idx < tools.length; k++, idx++) {
        phaseMap.set(idx, p);
      }
    }
    const lastP = phases[phases.length - 1];
    while (idx < tools.length) { phaseMap.set(idx, lastP); idx++; }
  }

  const _phase = (i) => {
    let p = tools[i].phase || phaseMap.get(i) || '_unassigned';
    if (p === 'entry') {
      p = phases.includes('plan') ? 'plan' : (phases.includes('act') ? 'act' : 'act');
    }
    return p;
  };

  let lastPhase = null;
  const closedPhases = new Set();

  for (let i = 0; i < tools.length; i++) {
    const tc = tools[i];
    const phase = _phase(i);

    // Phase transition
    if (phase !== lastPhase && phase !== '_unassigned') {
      // Close previous phase (emit thinking/review)
      if (lastPhase && !closedPhases.has(lastPhase)) {
        _emitPhaseClosing(events, conv, lastPhase, closedPhases, tools, _phase);
      }

      // Detect if this is the last phase in the stream
      let isLast = true;
      for (let j = i + 1; j < tools.length; j++) {
        const pj = _phase(j);
        if (pj !== phase && pj !== '_unassigned') { isLast = false; break; }
      }

      events.push({ type: 'phase_start', phase, isLast });
      lastPhase = phase;
    }

    // Classify and emit
    const cls = classifyToolForUserView(tc.name);
    if (cls !== 'hidden') {
      events.push({ type: cls, tc, id: String(i) });
    }
  }

  // Close last tool-bearing phase
  if (lastPhase && !closedPhases.has(lastPhase)) {
    _emitPhaseClosing(events, conv, lastPhase, closedPhases, tools, _phase);
  }

  // Handle phases that have content but no tools
  for (const p of stdPhases) {
    if (conv[p] && !closedPhases.has(p)) {
      const entry = conv[p];
      const content = entry.content ? String(entry.content) : '';
      const clean   = _cleanContent(content, p);

      events.push({ type: 'phase_start', phase: p, isLast: false });

      // Review: show ALL iterations from _history, not just the last
      if (p === 'review') {
        const history = entry._history || [];
        for (const prev of history) {
          const prevContent = prev.content ? String(prev.content) : '';
          if (prevContent.includes('REVIEW_FAILED')) {
            events.push({ type: 'review', passed: false });
          } else if (prevContent.includes('REVIEW_PASSED')) {
            events.push({ type: 'review', passed: true });
          }
        }
        // Current (latest) review
        if (content.includes('REVIEW_PASSED'))       events.push({ type: 'review', passed: true });
        else if (content.includes('REVIEW_FAILED'))  events.push({ type: 'review', passed: false });
      }

      // ACT content with no tools = prose answer
      if (p === 'act' && clean.trim()) {
        const hasActTools = tools.some((_, idx) => {
          const tp = _phase(idx);
          return tp === 'act';
        });
        if (!hasActTools) {
          events.push({ type: 'content', content: clean, phase: p });
          closedPhases.add(p);
          continue;
        }
      }

      if (clean.trim()) {
        events.push({ type: 'thinking', content: clean, phase: p });
      }
      closedPhases.add(p);
    }
  }

  // Inject LLM reasoning lines that weren't already emitted via thinking events
  // This picks up per-iteration reasoning from llm_calls.json
  for (const p of stdPhases) {
    const reasonings = llmReasoningByPhase[p] || [];
    if (reasonings.length > 0 && !closedPhases.has(`${p}_reasoning`)) {
      // Find the phase_start index for this phase
      const phaseStartIdx = events.findIndex(e => e.type === 'phase_start' && e.phase === p);
      if (phaseStartIdx >= 0) {
        // Insert reasoning lines right after the phase_start
        let insertIdx = phaseStartIdx + 1;
        for (const r of reasonings) {
          // Only add if it's concise reasoning (not the full phase summary)
          const trimmed = r.substring(0, 200);
          if (trimmed.length < 200 && trimmed.trim()) {
            events.splice(insertIdx, 0, { type: 'reasoning', content: trimmed, phase: p });
            insertIdx++;
          }
        }
      }
    }
  }

  return events;
}

/**
 * Emit thinking/review events for a phase that just ended (had tools).
 */
function _emitPhaseClosing(events, conv, phase, closedSet, tools, _phase) {
  closedSet.add(phase);
  const entry = conv[phase];
  if (!entry) return;

  const content = entry.content ? String(entry.content) : '';
  const clean   = _cleanContent(content, phase);

  // Review: show ALL iterations from _history
  if (phase === 'review') {
    const history = (entry._history || []);
    for (const prev of history) {
      const prevContent = prev.content ? String(prev.content) : '';
      if (prevContent.includes('REVIEW_FAILED')) events.push({ type: 'review', passed: false });
      else if (prevContent.includes('REVIEW_PASSED')) events.push({ type: 'review', passed: true });
    }
    if (content.includes('REVIEW_PASSED'))       events.push({ type: 'review', passed: true });
    else if (content.includes('REVIEW_FAILED'))  events.push({ type: 'review', passed: false });
  }

  // ACT content with no tools = prose answer
  if (phase === 'act' && clean.trim()) {
    const hasActTools = tools.some((_, idx) => {
      const tp = _phase(idx);
      return tp === 'act';
    });
    if (!hasActTools) {
      events.push({ type: 'content', content: clean, phase });
      return;
    }
  }

  // Thinking
  if (clean.trim()) {
    events.push({ type: 'thinking', content: clean, phase });
  }
}

// ---------------------------------------------------------------------------
// Event renderers — uniform > one-liner style
// ---------------------------------------------------------------------------

function _renderPhaseMarker(phase, isActive) {
  const label = PHASE_LABELS[phase] || `${phase}...`;
  const pulse = isActive ? '<span class="uv-pulse"></span>' : '';
  return `<div class="uv-evt-phase">${pulse}<span class="uv-evt-phase-label">\u25B8 ${esc(label)}</span></div>`;
}

function _renderDomainEvent(tc, id) {
  const summary = _getToolSummary(tc);
  const text = summary ? summary.text : tc.name;
  const ok = tc.success !== false;
  const badge = ok
    ? '<span class="uv-evt-badge uv-badge-ok">&#10003;</span>'
    : '<span class="uv-evt-badge uv-badge-fail">&#10007;</span>';
  const result = tc.result_content || tc.result || '';
  const preview = ok ? _extractPreview(result) : '';
  const prevHtml = preview ? `<span class="uv-evt-preview"> \u2014 ${esc(preview)}</span>` : '';

  let html = `<div class="uv-evt" data-bid="evt-${id}">`;
  html += `<div class="uv-evt-line" onclick="toggleBlock(this)">`;
  html += `<span class="uv-evt-gt">&gt;</span>`;
  html += `<span class="uv-evt-text">${esc(text)}${prevHtml}</span>`;
  html += badge;
  html += `</div>`;
  // Show inline error text for failed tool calls
  if (!ok && tc.error) {
    html += `<div class="uv-evt-error-line">${esc(tc.error)}</div>`;
  }
  html += _renderEvtDetail(tc);
  html += `</div>`;
  return html;
}

function _renderFindingEvent(tc, id) {
  if (tc.success === false) return '';
  const args = _parseArgs(tc.arguments);
  if (tc.name === 'batch_log_findings' && Array.isArray(args.findings)) {
    let html = '';
    for (let j = 0; j < args.findings.length; j++) {
      html += _renderSingleFinding(args.findings[j], `${id}-f${j}`);
    }
    return html;
  }
  return _renderSingleFinding(args, `${id}-f0`);
}

function _renderSingleFinding(args, id) {
  if (!args) return '';
  const content    = args.content || args.finding || args.detail || '';
  const category   = args.category || '';
  const confidence = args.confidence || '';
  const source     = args.source || '';
  if (!content) return '';

  const headerText = category
    ? `${category.toUpperCase()}: ${_truncate(content.replace(/\n/g, ' '), 60)}`
    : _truncate(content.replace(/\n/g, ' '), 70);

  let html = `<div class="uv-evt" data-bid="evt-${id}">`;
  html += `<div class="uv-evt-line" onclick="toggleBlock(this)">`;
  html += `<span class="uv-evt-gt">&gt;</span>`;
  html += `<span class="uv-evt-icon">\u25A0</span>`;
  html += `<span class="uv-evt-text">${esc(headerText)}</span>`;
  html += `</div>`;
  html += `<div class="uv-evt-detail">`;
  html += `<div class="uv-block-detail-content">${esc(content)}</div>`;
  if (source || confidence) {
    html += '<div class="uv-block-meta">';
    if (source)     html += `<span>${esc(source)}</span>`;
    if (confidence) html += `<span>confidence: ${esc(confidence)}</span>`;
    html += '</div>';
  }
  html += `</div></div>`;
  return html;
}

function _renderProgressEvent(tc, todoItems) {
  if (tc.success === false) return '';
  const args = _parseArgs(tc.arguments);
  if (tc.name === 'batch_mark_todo_complete' && Array.isArray(args.items)) {
    let html = '';
    for (const item of args.items) {
      const idx  = Number(item.item_index ?? item.index);
      const desc = _resolveTodoDescription(idx, todoItems) || item.summary || `item ${idx}`;
      html += `<div class="uv-evt-phase"><span class="uv-evt-gt uv-line-ok">\u2713</span><span>Completed: ${esc(desc)}</span></div>`;
    }
    return html;
  }
  const idx  = Number(args.item_index);
  const desc = _resolveTodoDescription(idx, todoItems) || args.summary || `item ${idx}`;
  return `<div class="uv-evt-phase"><span class="uv-evt-gt uv-line-ok">\u2713</span><span>Completed: ${esc(desc)}</span></div>`;
}

function _renderSpawnEvent(tc, id) {
  const args = _parseArgs(tc.arguments);
  const role = args.role || args.agent_role || 'agent';
  const task = args.task || args.objective || '';
  const taskText = task ? ` \u2014 "${_truncate(task, 50)}"` : '';

  let html = `<div class="uv-evt" data-bid="evt-${id}">`;
  html += `<div class="uv-evt-line" onclick="toggleBlock(this)">`;
  html += `<span class="uv-evt-gt">&gt;</span>`;
  html += `<span class="uv-evt-icon">+</span>`;
  html += `<span class="uv-evt-text">Spawned: ${esc(role)}${esc(taskText)}</span>`;
  html += `</div>`;
  html += _renderEvtDetail(tc);
  html += `</div>`;
  return html;
}

function _renderWaitEvent(tc, id, subAgents) {
  const count = Object.keys(subAgents || {}).length;
  const label = count > 0
    ? `Waiting for ${count} sub-agent${count > 1 ? 's' : ''}...`
    : 'Waiting for sub-agents...';

  let html = `<div class="uv-evt" data-bid="evt-${id}">`;
  html += `<div class="uv-evt-line" onclick="toggleBlock(this)">`;
  html += `<span class="uv-evt-gt">&gt;</span>`;
  html += `<span class="uv-evt-icon">\u2026</span>`;
  html += `<span class="uv-evt-text">${esc(label)}</span>`;
  html += `</div>`;
  const result = tc.result_content || tc.result || '';
  if (result) {
    html += `<div class="uv-evt-detail">`;
    const resStr = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
    html += `<pre class="uv-block-pre">${esc(resStr)}</pre>`;
    html += `</div>`;
  }
  html += `</div>`;
  return html;
}

function _renderThinkingEvent(content, phase, tid) {
  if (!content || !content.trim()) return '';
  const id = `think-${tid}-${phase}`;
  let html = `<div class="uv-evt uv-evt-thinking" data-bid="evt-${id}">`;
  html += `<div class="uv-evt-line" onclick="toggleBlock(this)">`;
  html += `<span class="uv-evt-gt">&gt;</span>`;
  html += `<span class="uv-evt-text uv-evt-think-label">Agent's reasoning</span>`;
  html += `</div>`;
  html += `<div class="uv-evt-detail">`;
  html += `<div class="uv-block-detail-content uv-block-mono">${renderMarkdown(content)}</div>`;
  html += `</div></div>`;
  return html;
}

function _renderReasoningLine(content) {
  if (!content || !content.trim()) return '';
  return `<div class="uv-evt-reasoning"><span class="uv-evt-reason-icon">\u25B7</span><span>${esc(_truncate(content, 150))}</span></div>`;
}

function _renderReviewBadge(passed) {
  if (passed) {
    return `<div class="uv-evt-phase"><span class="uv-evt-gt uv-line-ok">\u2713</span><span>Review passed</span></div>`;
  }
  return `<div class="uv-evt-phase"><span class="uv-evt-gt uv-line-fail">\u2717</span><span>Review flagged issues</span></div>`;
}

/** Shared detail block for tool events (input/output/error). */
function _renderEvtDetail(tc) {
  let html = `<div class="uv-evt-detail">`;
  if (tc.arguments) {
    const argStr = typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments, null, 2);
    if (argStr && argStr !== '{}') {
      html += `<div class="uv-block-section">Input</div><pre class="uv-block-pre">${esc(argStr)}</pre>`;
    }
  }
  const result = tc.result_content || tc.result || '';
  if (result) {
    const resStr = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
    html += `<div class="uv-block-section">Output</div><pre class="uv-block-pre">${esc(resStr)}</pre>`;
  }
  if (tc.error) {
    html += `<div class="uv-block-section uv-block-error">Error</div><pre class="uv-block-pre uv-block-error">${esc(tc.error)}</pre>`;
  }
  html += `</div>`;
  return html;
}

// ---------------------------------------------------------------------------
// Plan checklist (unchanged — always visible)
// ---------------------------------------------------------------------------
function _renderPlanChecklist(items, done, isLive) {
  const firstPending = items.find(t => !t.completed);
  let html = '<div class="uv-plan">';
  html += '<div class="uv-plan-header">';
  html += `<span class="uv-plan-label">Plan</span>`;
  html += `<span class="uv-plan-count">${done}/${items.length}</span>`;
  html += '</div><div class="uv-plan-items">';
  for (const item of items) {
    const isDone    = item.completed;
    const isCurrent = !isDone && isLive && item === firstPending;
    const icon = isDone ? '&#10003;' : '&#9675;';
    const cls  = isDone ? ' uv-plan-done' : (isCurrent ? ' uv-plan-current' : '');
    html += `<div class="uv-plan-item${cls}"><span class="uv-plan-check">${icon}</span><span>${esc(item.description)}</span></div>`;
  }
  html += '</div></div>';
  return html;
}

// ---------------------------------------------------------------------------
// Status indicator (unchanged)
// ---------------------------------------------------------------------------
function _renderStatusIndicator(status, activity) {
  const st = String(status).toLowerCase();
  if (st === 'running') {
    const doing = activity.current_doing || 'Working';
    return `<span class="uv-status uv-status-running"><span class="uv-pulse"></span>${esc(doing)}</span>`;
  }
  if (st === 'completed')  return `<span class="uv-status uv-status-done">&#10003; Done</span>`;
  if (st === 'failed' || st === 'error') return `<span class="uv-status uv-status-failed">&#10007; Failed</span>`;
  if (st === 'cancelled')  return `<span class="uv-status uv-status-cancelled">Cancelled</span>`;
  return `<span class="uv-status uv-status-idle">${esc(st)}</span>`;
}

// ---------------------------------------------------------------------------
// Error banner (unchanged)
// ---------------------------------------------------------------------------
function _renderErrorBanner(agent) {
  const output = agent.output || {};
  const errors = output.errors || [];
  const retryBtn = state.config?.can_continue
    ? ` <button class="uv-error-retry" onclick="retryFailedSession('${esc((agent.info || {}).task_id || '')}')">Retry</button>`
    : '';

  // Structured error display when we have multiple errors
  if (errors.length > 1) {
    let html = `<div class="uv-error"><span class="uv-error-icon">&#9888;</span><div class="uv-error-list">`;
    for (const e of errors) {
      const msg = e.message || e.error || JSON.stringify(e);
      const source = e.error_type || e.type || '';
      const sourceTag = source
        ? `<span class="uv-error-source">${esc(source)}</span>`
        : '';
      html += `<div class="uv-error-item">${sourceTag}<span>${esc(msg)}</span></div>`;
    }
    html += `</div>${retryBtn}</div>`;
    return html;
  }

  // Single error or fallback
  let errorMsg = '';
  if (errors.length === 1) {
    const e = errors[0];
    errorMsg = e.message || e.error || JSON.stringify(e);
  }
  if (!errorMsg && agent._parent_failure_context) {
    errorMsg = agent._parent_failure_context;
  }
  const saSummary = agent.sub_agents_summary || [];
  if (!errorMsg && saSummary.length) {
    const failedSa = saSummary.filter(s => s.status === 'failed');
    if (failedSa.length) errorMsg = `${failedSa.length} sub-agent(s) failed`;
  }
  if (!errorMsg) errorMsg = 'Something went wrong';
  return `<div class="uv-error"><span class="uv-error-icon">&#9888;</span> <span>${esc(errorMsg)}</span>${retryBtn}</div>`;
}

// ---------------------------------------------------------------------------
// Content cleaning (unchanged)
// ---------------------------------------------------------------------------
function _cleanContent(content, phase) {
  let text = content;
  text = text.replace(/^Plan Summary:\s*/i, '');
  text = text.replace(/^Execution Summary:\s*/i, '');
  text = text.replace(/^Review Summary:\s*/i, '');
  text = text.replace(/^Report Summary:\s*/i, '');
  return text;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _getToolSummary(tc) {
  const gen = TOOL_SUMMARIES[tc.name];
  if (!gen) return null;
  return gen(_parseArgs(tc.arguments));
}

function _parseArgs(args) {
  if (!args) return {};
  if (typeof args === 'string') {
    try { return JSON.parse(args); } catch { return {}; }
  }
  return args;
}

function _resolveTodoDescription(index, todoItems) {
  if (!Number.isFinite(index) || !todoItems.length) return null;
  const item = todoItems.find(t => t.index === index);
  return item ? item.description : null;
}

function _hasRenderableValue(value) {
  if (value === null || value === undefined) return false;
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value).length > 0;
  return true;
}

function _truncate(text, max) {
  if (!text || text.length <= max) return text;
  return text.substring(0, max) + '...';
}

/**
 * Extract a user-friendly preview from a tool result.
 * Tries to parse JSON and extract title/content fields, falls back to first text line.
 */
function _extractPreview(result) {
  if (!result) return '';
  let str = String(result).trim();
  if (!str) return '';

  // Strip XML-like wrapper tags
  str = str.replace(/^<[a-z_]+>\s*/i, '').replace(/\s*<\/[a-z_]+>\s*$/i, '').trim();

  if (str.startsWith('{') || str.startsWith('[')) {
    try {
      const obj = JSON.parse(str);
      if (obj.title)   return _truncate(obj.title, 60);
      if (obj.name)    return _truncate(obj.name, 60);
      if (obj.summary) return _truncate(obj.summary, 60);
      if (obj.content) return _truncate(String(obj.content).split('\n')[0], 60);
      if (obj.full_text) return _truncate(String(obj.full_text).split('\n')[0], 60);
      if (obj.text)    return _truncate(String(obj.text).split('\n')[0], 60);
      if (Array.isArray(obj) && obj.length) return `${obj.length} results`;
      if (Array.isArray(obj.results) && obj.results.length) return `${obj.results.length} results`;
      if (typeof obj.count === 'number') return `${obj.count} results`;
      if (typeof obj.total === 'number') return `${obj.total} results`;
    } catch {
      const titleMatch = str.match(/"title"\s*:\s*"([^"]+)"/);
      if (titleMatch) return _truncate(titleMatch[1], 60);
      const nameMatch = str.match(/"name"\s*:\s*"([^"]+)"/);
      if (nameMatch) return _truncate(nameMatch[1], 60);
      return '';
    }
  }

  // Plain text: first non-empty, non-tag line
  const lines = str.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith('<') && trimmed.endsWith('>')) continue;
    return _truncate(trimmed, 60);
  }
  return '';
}
