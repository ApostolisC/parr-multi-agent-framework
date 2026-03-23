/* Chat view — continuous stream renderer (recursive). */
import { esc, fmtNum } from './utils.js';
import { renderMarkdown } from './renderers.js';
import { renderStructuredReport } from './report.js';

export function renderChatView(agent, tid) {
  return _renderChatStream(agent, tid, 0);
}

function _friendlyPhaseLabel(phase) {
  if (!phase) return 'Plan';
  const s = String(phase);
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function _resolveActivityState(activity, status) {
  const st2 = String((activity || {}).state || '').toLowerCase();
  if (st2) return st2;
  const st = String(status || '').toLowerCase();
  if (st === 'running') return 'working';
  if (st === 'spawned' || st === 'queued') return 'waiting';
  if (st === 'completed') return 'done';
  if (st === 'failed' || st === 'error' || st === 'cancelled') return 'failed';
  return 'idle';
}

function _resolveCurrentDoing(activity, status, currentPhase) {
  const doing = (activity || {}).current_doing;
  if (doing) return String(doing);
  const st = String(status || '').toLowerCase();
  if (st === 'running') return `executing ${currentPhase.toLowerCase()} phase`;
  if (st === 'completed') return 'completed';
  if (st === 'failed' || st === 'error') return 'failed';
  return st || 'idle';
}

function _activityClassName(s) {
  if (s === 'working') return 'cs-activity-working';
  if (s === 'waiting') return 'cs-activity-waiting';
  if (s === 'done') return 'cs-activity-done';
  if (s === 'failed') return 'cs-activity-failed';
  return 'cs-activity-idle';
}

function _renderShimmerText(text) {
  const raw = String(text || '');
  if (!raw) return '';
  let html = '<span class="cs-shimmer">';
  let delay = 0;
  for (const ch of raw) {
    if (ch === ' ') {
      html += '<span class="cs-shimmer-space">&nbsp;</span>';
      continue;
    }
    html += `<span class="cs-shimmer-char" style="animation-delay:${delay.toFixed(2)}s">${esc(ch)}</span>`;
    delay += 0.05;
  }
  html += '</span>';
  return html;
}

function _renderAnimatedText(text, animate) {
  const raw = String(text || '');
  if (!raw) return '';
  return animate ? _renderShimmerText(raw) : esc(raw);
}

function _hasRenderableValue(value) {
  if (value === null || value === undefined) return false;
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value).length > 0;
  return true;
}

function _normalizeTodoItem(item, fallbackIndex) {
  const raw = item || {};
  const idx = Number(raw.index);
  return {
    index: Number.isFinite(idx) ? idx : fallbackIndex,
    description: String(raw.description || ''),
    priority: String(raw.priority || 'medium'),
    completed: !!raw.completed,
    completion_summary: raw.completion_summary ? String(raw.completion_summary) : '',
  };
}

function _parseTodoListText(rawText) {
  const text = String(rawText || '').trim();
  if (!text) return null;
  if (text.includes('No todo items.')) return [];
  const parsed = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const m = line.match(/^\[(x| )\]\s*(\d+)\.\s*\[([^\]]+)\]\s*(.+)$/i);
    if (!m) continue;
    let description = m[4].trim();
    let completionSummary = '';
    const summaryParts = description.split(/\s+(?:—|-|—)\s+/);
    if (summaryParts.length > 1) {
      description = summaryParts[0].trim();
      completionSummary = summaryParts.slice(1).join(' - ').trim();
    }
    parsed.push({
      index: Number(m[2]),
      description,
      priority: m[3].trim() || 'medium',
      completed: m[1].toLowerCase() === 'x',
      completion_summary: completionSummary,
    });
  }
  return parsed.length ? parsed : null;
}

export function _buildLiveTodoList(agent) {
  const memory = agent.memory || {};
  const memoryTodo = Array.isArray(memory.todo_list) ? memory.todo_list : [];
  const toolCalls = Array.isArray(agent.tool_calls) ? agent.tool_calls : [];

  const todoToolNames = new Set([
    'create_todo_list',
    'update_todo_list',
    'get_todo_list',
    'mark_todo_complete',
    'batch_mark_todo_complete',
  ]);
  const todoCalls = toolCalls.filter(tc => todoToolNames.has(tc.name));

  if (!todoCalls.length) {
    return memoryTodo.map((item, i) => _normalizeTodoItem(item, i)).sort((a, b) => a.index - b.index);
  }

  const todoMap = new Map();
  const hasCreate = todoCalls.some(tc => tc.name === 'create_todo_list' && tc.success !== false);

  function upsert(item, fallbackIndex) {
    const normalized = _normalizeTodoItem(item, fallbackIndex);
    todoMap.set(normalized.index, normalized);
  }

  function maxTodoIndex() {
    if (!todoMap.size) return -1;
    return Math.max(...Array.from(todoMap.keys()));
  }

  if (!hasCreate) {
    memoryTodo.forEach((item, i) => upsert(item, i));
  }

  for (const tc of todoCalls) {
    if (!tc || tc.success === false) continue;
    const args = tc.arguments || {};

    if (tc.name === 'create_todo_list') {
      todoMap.clear();
      const items = Array.isArray(args.items) ? args.items : [];
      items.forEach((item, i) => upsert({
        index: i,
        description: item.description || '',
        priority: item.priority || 'medium',
        completed: false,
        completion_summary: '',
      }, i));
      continue;
    }

    if (tc.name === 'update_todo_list') {
      const remove = Array.isArray(args.remove) ? args.remove : [];
      remove.forEach(idx => {
        const n = Number(idx);
        if (Number.isFinite(n)) todoMap.delete(n);
      });

      const modify = Array.isArray(args.modify) ? args.modify : [];
      modify.forEach((mod, i) => {
        const n = Number(mod.index ?? mod.item_index);
        if (!Number.isFinite(n)) return;
        const existing = todoMap.get(n) || _normalizeTodoItem({ index: n }, i);
        if (mod.description !== undefined) existing.description = String(mod.description || '');
        if (mod.priority !== undefined) existing.priority = String(mod.priority || 'medium');
        if (mod.completed !== undefined) existing.completed = !!mod.completed;
        if (mod.completion_summary !== undefined) existing.completion_summary = String(mod.completion_summary || '');
        todoMap.set(n, existing);
      });

      if (Number.isFinite(Number(args.item_index))) {
        const n = Number(args.item_index);
        const existing = todoMap.get(n) || _normalizeTodoItem({ index: n }, n);
        if (args.description !== undefined) existing.description = String(args.description || '');
        if (args.priority !== undefined) existing.priority = String(args.priority || 'medium');
        todoMap.set(n, existing);
      }

      const add = Array.isArray(args.add) ? args.add : [];
      if (add.length) {
        let idx = maxTodoIndex() + 1;
        add.forEach(item => {
          upsert({
            index: idx,
            description: item.description || '',
            priority: item.priority || 'medium',
            completed: false,
            completion_summary: '',
          }, idx);
          idx += 1;
        });
      }
      continue;
    }

    if (tc.name === 'mark_todo_complete') {
      const n = Number(args.item_index);
      if (Number.isFinite(n)) {
        const existing = todoMap.get(n) || _normalizeTodoItem({ index: n }, n);
        existing.completed = true;
        existing.completion_summary = String(args.summary || existing.completion_summary || '');
        todoMap.set(n, existing);
      }
      continue;
    }

    if (tc.name === 'batch_mark_todo_complete') {
      const items = Array.isArray(args.items) ? args.items : [];
      items.forEach((entry, i) => {
        const n = Number(entry.item_index ?? entry.index);
        if (!Number.isFinite(n)) return;
        const existing = todoMap.get(n) || _normalizeTodoItem({ index: n }, i);
        existing.completed = true;
        existing.completion_summary = String(entry.summary || existing.completion_summary || '');
        todoMap.set(n, existing);
      });
      continue;
    }

    if (tc.name === 'get_todo_list') {
      const parsed = _parseTodoListText(tc.result_content || tc.result || '');
      if (Array.isArray(parsed)) {
        todoMap.clear();
        parsed.forEach((item, i) => upsert(item, i));
      }
    }
  }

  return Array.from(todoMap.values()).sort((a, b) => a.index - b.index);
}

function _renderInlineTodoList(items) {
  if (!Array.isArray(items) || !items.length) return '';
  const completed = items.filter(item => item.completed).length;
  let html = `<div class="cs-todo-inline"><div class="cs-todo-title">Todo ${completed}/${items.length}</div>`;
  const visible = items.slice(0, 10);
  for (const item of visible) {
    const done = !!item.completed;
    html += `<div class="cs-todo-item${done ? ' done' : ''}">`;
    html += `<span class="cs-todo-icon ${done ? 'done' : 'pending'}">${done ? '&#10003;' : '&#9675;'}</span>`;
    html += `<span class="cs-todo-text">${esc(item.description || '')}</span>`;
    if (item.completion_summary) {
      html += `<span class="cs-todo-summary">- ${esc(item.completion_summary)}</span>`;
    }
    html += `</div>`;
  }
  if (items.length > visible.length) {
    html += `<div class="text-xs text-muted">+${items.length - visible.length} more...</div>`;
  }
  html += '</div>';
  return html;
}

/**
 * Recursive stream renderer. `depth` controls whether this is root (0)
 * or a nested sub-agent.
 */
function _renderChatStream(agent, tid, depth) {
  const conv = agent.conversation || {};
  const tools = agent.tool_calls || [];
  const subAgents = agent.sub_agents || {};
  const info = agent.info || {};
  const output = agent.output || {};
  const status = agent._workflow_status || info.status || output.status || 'unknown';
  const metrics = agent.metrics || {};
  const tokens = metrics.tokens || {};
  const context = metrics.context || {};
  const activity = metrics.activity || {};
  const phaseOutputs = (output.execution_metadata || {}).phase_outputs || {};
  const phaseOrder = ['plan', 'act', 'review', 'report'];
  const phases = phaseOrder.filter(p => conv[p] || phaseOutputs[p]);
  const finalReport = _hasRenderableValue(output.findings)
    ? output.findings
    : (_hasRenderableValue(phaseOutputs.report) ? phaseOutputs.report : phaseOutputs.act);
  const hasFinalReport = _hasRenderableValue(finalReport);

  const activityState = _resolveActivityState(activity, status);
  const currentPhaseKey = String(activity.current_phase || phases[phases.length - 1] || 'plan').toLowerCase();
  const currentPhase = _friendlyPhaseLabel(currentPhaseKey);
  const currentDoing = _resolveCurrentDoing(activity, status, currentPhase);
  const isLiveActivity = activityState === 'working' || activityState === 'waiting';
  const todoItems = _buildLiveTodoList(agent);
  const todoDone = todoItems.filter(t => t.completed).length;

  // Distribute tools to phases
  const hasPhaseField = tools.some(t => t.phase);
  const toolsByPhase = {};
  if (hasPhaseField) {
    for (const t of tools) {
      const p = t.phase || '_unassigned';
      (toolsByPhase[p] = toolsByPhase[p] || []).push(t);
    }
  } else {
    let idx = 0;
    for (const p of phases) {
      const count = (conv[p] || {}).tool_calls_count || 0;
      toolsByPhase[p] = tools.slice(idx, idx + count);
      idx += count;
    }
    if (idx < tools.length && phases.length) {
      const last = phases[phases.length - 1];
      toolsByPhase[last] = (toolsByPhase[last] || []).concat(tools.slice(idx));
    }
  }

  let html = '<div class="chat-stream">';

  html += `<div class="cs-chat-head">`;
  html += `<div class="cs-chat-head-main">`;
  html += `<span class="cs-agent-name">${esc(info.role || 'agent')}${esc(info.sub_role ? ` / ${info.sub_role}` : '')}</span>`;
  html += `<span class="cs-activity-badge ${_activityClassName(activityState)}">${esc(activityState)}</span>`;
  html += `<span class="text-xs text-muted">${_renderAnimatedText(`phase: ${currentPhase}`, isLiveActivity)}</span>`;
  html += `<span class="cs-current-doing">${_renderAnimatedText(`current doing: ${currentDoing}`, isLiveActivity)}</span>`;
  html += `</div>`;
  html += `<div class="cs-chat-metrics">`;
  html += `<span>tokens: ${fmtNum(tokens.total || 0)}${tokens.is_estimated ? ' est' : ''}</span>`;
  html += `<span>context: ${fmtNum(context.estimated_tokens || 0)} tok</span>`;
  html += `<span>todo: ${todoDone}/${todoItems.length}</span>`;
  html += `</div>`;
  html += _renderInlineTodoList(todoItems);
  html += `</div>`;

  if (!phases.length && status !== 'failed' && !hasFinalReport) {
    html += '<div class="placeholder">No conversation data yet.</div>';
    html += '</div>';
    return html;
  }

  for (const phase of phases) {
    const entry = conv[phase] || {};
    const phaseTools = toolsByPhase[phase] || [];
    const phaseLabel = _friendlyPhaseLabel(phase);
    const isActivePhase = isLiveActivity && phase.toLowerCase() === currentPhaseKey;
    const iterTxt = entry.iterations ? `${entry.iterations} iter` : '';
    const limitHit = entry.hit_iteration_limit;
    const phaseId = `ph-${tid}-${phase}`;
    const toolCount = phaseTools.length;
    const phaseSummary = [iterTxt, toolCount ? `${toolCount} tools` : ''].filter(Boolean).join(', ');

    // Collapsible phase wrapper
    html += `<div class="cs-phase-wrap" data-phid="${phaseId}">`;
    html += `<div class="cs-phase-toggle" onclick="togglePhaseSection(this)">`;
    html += `<span class="cs-chevron">&#9654;</span>`;
    html += `<span class="cs-phase-label-inline">${_renderAnimatedText(phaseLabel, isActivePhase)}</span>`;
    if (phaseSummary) html += `<span class="cs-phase-summary">${esc(phaseSummary)}</span>`;
    if (limitHit) html += `<span class="tag tag-red">limit</span>`;
    html += `</div>`;

    // Phase content (hidden by default, shown when .open)
    html += `<div class="cs-phase-content">`;

    // Original phase rendering inside
    html += `<div class="cs-phase">`;
    html += `<span class="cs-phase-label">${_renderAnimatedText(phaseLabel, isActivePhase)}</span>`;
    if (iterTxt || limitHit) {
      html += `<div class="cs-phase-meta">${esc(iterTxt)}${limitHit ? ' <span class="tag tag-red">limit</span>' : ''}</div>`;
    }

    html += `<div class="cs-phase-body">`;

    // LLM content
    if (entry.content) {
      html += `<div class="cs-agent">${renderMarkdown(entry.content)}</div>`;
    }

    // Tool calls - compact collapsed rows
    for (let i = 0; i < phaseTools.length; i++) {
      const tc = phaseTools[i];
      const tcId = `cst-${tid}-${phase}-${i}`;
      const ok = tc.success !== false;
      const badge = ok
        ? '<span class="cs-tool-badge tag tag-ok">ok</span>'
        : '<span class="cs-tool-badge tag tag-fail">fail</span>';

      html += `<div class="cs-tool" data-cstid="${tcId}">`;
      html += `<div class="cs-tool-row" onclick="toggleToolRow(this)">`;
      html += `<span class="cs-chevron">&#9654;</span>`;
      html += `<span class="cs-tool-icon">&#9889;</span>`;
      html += `<span class="cs-tool-name">${esc(tc.name)}</span>`;
      html += badge;
      html += `</div>`;

      // Expandable detail
      html += `<div class="cs-tool-detail">`;
      if (tc.arguments) {
        const argStr = typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments, null, 2);
        if (argStr && argStr !== '{}') {
          html += `<div class="cs-tool-section">Input</div>${esc(argStr)}\n`;
        }
      }
      const result = tc.result_content || tc.result;
      if (result) {
        const resStr = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
        html += `<div class="cs-tool-section">Output</div>${esc(resStr)}\n`;
      }
      if (tc.error) {
        html += `<div class="cs-tool-section cs-tool-error">Error</div><span class="cs-tool-error">${esc(tc.error)}</span>\n`;
      }
      html += `</div></div>`;
    }

    // Sub-agent spawns inline after act phase
    if (phase === 'act' && Object.keys(subAgents).length) {
      html += _renderInlineSubAgents(subAgents, tid, depth, output.errors);
    }

    html += `</div></div>`; // close cs-phase-body, cs-phase
    html += `</div></div>`; // close cs-phase-content, cs-phase-wrap
  }

  // If sub-agents exist but no 'act' phase was reached, show them after the last phase
  const subKeys = Object.keys(subAgents);
  const shownInAct = phases.includes('act') && subKeys.length;
  if (!shownInAct && subKeys.length) {
    html += _renderInlineSubAgents(subAgents, tid, depth, output.errors);
  }

  // Error / failure banner
  const isFailed = status === 'failed' || status === 'error';
  if (isFailed) {
    let errorMsg = '';
    // 1. Check output.errors (most detailed)
    const errors = output.errors || [];
    if (errors.length) {
      errorMsg = errors.map(e => e.message || e.error || JSON.stringify(e)).join('; ');
    }
    // 2. Check parent-injected failure context (set by _renderInlineSubAgents)
    if (!errorMsg && agent._parent_failure_context) {
      errorMsg = agent._parent_failure_context;
    }
    // 3. Check sub_agents_summary for child failures
    const saSummary = agent.sub_agents_summary || [];
    if (!errorMsg && saSummary.length) {
      const failedSa = saSummary.filter(s => s.status === 'failed');
      if (failedSa.length) {
        errorMsg = `${failedSa.length} sub-agent(s) failed`;
      }
    }
    // 4. Generic fallback
    if (!errorMsg) {
      errorMsg = 'Agent failed - no error details available';
    }
    html += `<div class="cs-error"><span class="cs-error-icon">&#9888;</span> ${esc(errorMsg)}</div>`;
  }

  // Final output / report (prefer submitted structured report over phase summary text)
  if (hasFinalReport) {
    const renderedFinal = (typeof finalReport === 'object' && finalReport !== null)
      ? renderStructuredReport(finalReport)
      : renderMarkdown(typeof finalReport === 'string' ? finalReport : JSON.stringify(finalReport, null, 2));
    html += `<div class="cs-phase"><span class="cs-phase-label">${esc('Output')}</span>`;
    html += `<div class="cs-phase-body"><div class="cs-output">${renderedFinal}</div></div></div>`;
  }

  html += '</div>';
  return html;
}

/**
 * Render sub-agent spawn blocks. Used both inside act-phase and as fallback.
 * Injects _parent_failure_context into each child so error banners are specific.
 */
function _renderInlineSubAgents(subAgents, tid, depth, parentErrors) {
  let html = '';
  // Extract specific limit type from parent errors for child failure context
  let limitDetail = 'budget';
  if (parentErrors && parentErrors.length) {
    for (const err of parentErrors) {
      const msg = err.message || err.error || '';
      const m = msg.match(/^\[(tokens|cost|duration)\]/);
      if (m) { limitDetail = m[1]; break; }
    }
  }
  for (const [saId, sa] of Object.entries(subAgents)) {
    const saInfo = sa.info || {};
    const saRole = saInfo.role || saId;
    const saSubRole = saInfo.sub_role ? ` / ${saInfo.sub_role}` : '';
    const saTask = saInfo.task || '';
    const saStatus = saInfo.status || 'unknown';
    const saActivity = ((sa.metrics || {}).activity || {});
    const saState = _resolveActivityState(saActivity, saStatus);
    const saPhase = _friendlyPhaseLabel(saActivity.current_phase || '');
    const saBadge = saState === 'done'
      ? '<span class="tag tag-green">done</span>'
      : saState === 'working'
        ? '<span class="tag tag-blue">working</span>'
        : saState === 'waiting'
          ? '<span class="tag tag-warn">waiting</span>'
          : saState === 'failed'
            ? '<span class="tag tag-red">failed</span>'
            : '<span class="tag tag-dim">idle</span>';
    const spawnId = `css-${tid}-${saId}`;

    // Inject parent-level failure context for the child's error banner
    if (saStatus === 'failed' && !(sa.output || {}).errors) {
      sa._parent_failure_context = `Terminated - parent workflow ${limitDetail} limit exceeded`;
    }

    html += `<div class="cs-spawn" data-cssid="${spawnId}">`;
    html += `<div class="cs-spawn-row" onclick="toggleSpawnRow(this)">`;
    html += `<span class="cs-chevron">&#9654;</span>`;
    html += `<span class="cs-spawn-icon">&#8627;</span>`;
    html += `<span style="font-weight:600">${esc(saRole)}${esc(saSubRole)}</span>`;
    html += saBadge;
    if (saPhase) html += `<span class="tag tag-dim">${esc(saPhase)}</span>`;
    html += `</div>`;

    // Expanded detail: task assignment + nested stream
    html += `<div class="cs-spawn-detail">`;
    if (saTask) {
      html += `<div class="cs-task-assignment"><span class="cs-task-label">Assigned:</span> ${esc(saTask)}</div>`;
    }
    html += _renderChatStream(sa, `${tid}-sa-${saId}`, depth + 1);
    html += `</div></div>`;
  }
  return html;
}
