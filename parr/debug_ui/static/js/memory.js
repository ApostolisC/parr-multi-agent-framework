/* Memory, todo list, findings, and review renderers. */
import { esc, formatJson } from './utils.js';

export function renderMemory(mem, tid) {
  let html = '';
  for (const [key, val] of Object.entries(mem)) {
    if (val === null || val === undefined) continue;
    const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const count = Array.isArray(val) ? ` (${val.length})` : '';

    html += `<details class="collapse mb-8" data-cid="mem-${tid}-${key}">
      <summary><span class="text-sm">${esc(label)}${count}</span></summary>
      <div class="collapse-body">`;

    if (key === 'todo_list' && Array.isArray(val)) {
      html += renderTodoList(val);
    } else if (key === 'findings' && Array.isArray(val)) {
      html += renderFindings(val);
    } else if (key === 'review' && Array.isArray(val)) {
      html += renderReview(val);
    } else {
      html += `<div class="json-block">${esc(formatJson(val))}</div>`;
    }

    html += `</div></details>`;
  }
  return html;
}

export function renderTodoList(items) {
  if (!items.length) return '<span class="text-muted">Empty</span>';
  let html = '<div style="display:flex;flex-direction:column;gap:4px">';
  for (const t of items) {
    const done = t.completed;
    const icon = done ? '&#10003;' : '&#9675;';
    const style = done ? 'color:var(--green)' : 'color:var(--text-dim)';
    html += `<div class="flex items-center gap-6">
      <span style="${style};font-weight:600;width:14px">${icon}</span>
      <span class="${done ? 'text-dim' : ''}">${esc(t.description || '')}</span>
      ${t.completion_summary ? `<span class="text-xs text-muted">- ${esc(t.completion_summary)}</span>` : ''}
    </div>`;
  }
  html += '</div>';
  return html;
}

export function renderFindings(items) {
  if (!items.length) return '<span class="text-muted">Empty</span>';
  let html = '';
  for (const f of items) {
    html += `<div style="margin-bottom:8px;padding:8px;background:var(--bg-input);border-radius:var(--radius-sm);border-left:3px solid var(--accent-dim)">
      <div class="flex items-center gap-6 mb-8">
        <span class="tag tag-tool">${esc(f.category || 'general')}</span>
        ${f.confidence ? `<span class="text-xs text-muted">${esc(f.confidence)}</span>` : ''}
      </div>
      <div class="text-sm">${esc(f.content || '')}</div>
      ${f.source ? `<div class="text-xs text-muted mt-8">Source: ${esc(f.source)}</div>` : ''}
    </div>`;
  }
  return html;
}

export function renderReview(items) {
  if (!items.length) return '<span class="text-muted">Empty</span>';
  let html = '';
  for (const r of items) {
    const ratingClass = r.rating === 'pass' ? 'tag-ok' : (r.rating === 'fail' ? 'tag-fail' : 'tag-warn');
    html += `<div style="margin-bottom:6px" class="flex items-center gap-6">
      <span class="tag ${ratingClass}">${esc(r.rating || '?')}</span>
      <span class="text-sm">${esc(r.criterion || '')}</span>
      ${r.justification ? `<span class="text-xs text-muted">- ${esc(r.justification)}</span>` : ''}
    </div>`;
  }
  return html;
}
