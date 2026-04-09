/* Structured report renderer. */
import { esc, formatJson } from './utils.js';
import { renderMarkdown } from './renderers.js';

export function renderStructuredReport(report) {
  if (!report || typeof report !== 'object') {
    return `<div class="json-block">${esc(formatJson(report))}</div>`;
  }

  let html = '';
  if (report.answer) {
    html += `<div class="mb-8"><strong class="text-xs text-muted">Answer</strong>`;
    html += `<div class="agent-task">${renderMarkdown(String(report.answer))}</div></div>`;
  }

  if (report.summary) {
    html += `<div class="mb-8"><strong class="text-xs text-muted">Summary</strong>`;
    html += `<div class="agent-task">${esc(report.summary)}</div></div>`;
  }

  if (Array.isArray(report.key_findings) && report.key_findings.length) {
    html += '<div class="mb-8"><strong class="text-xs text-muted">Key Findings</strong>';
    html += '<ol style="padding-left:18px;margin-top:6px">';
    for (const item of report.key_findings) {
      if (!item) continue;
      const finding = item.finding || item.detail || item.content || formatJson(item);
      const source = item.source ? `<div class="text-xs text-muted mt-8">Source: ${esc(item.source)}</div>` : '';
      const evidence = item.evidence ? `<div class="text-xs text-muted mt-8">Evidence: ${esc(item.evidence)}</div>` : '';
      const confidence = item.confidence ? `<div class="text-xs text-muted mt-8">Confidence: ${esc(item.confidence)}</div>` : '';
      html += `<li class="text-sm mb-8">${esc(finding)}${source}${evidence}${confidence}</li>`;
    }
    html += '</ol></div>';
  }

  if (Array.isArray(report.evidence) && report.evidence.length) {
    html += '<div class="mb-8"><strong class="text-xs text-muted">Evidence</strong>';
    html += '<ul style="padding-left:16px;margin-top:6px">';
    for (const item of report.evidence) {
      if (!item) continue;
      if (typeof item === 'string') {
        html += `<li class="text-sm mb-8">${esc(item)}</li>`;
        continue;
      }
      const detail = item.detail || item.content || item.finding || formatJson(item);
      const source = item.source ? `<div class="text-xs text-muted mt-8">Source: ${esc(item.source)}</div>` : '';
      const url = item.url ? `<div class="text-xs text-muted mt-8">URL: ${esc(item.url)}</div>` : '';
      const quote = item.quote ? `<div class="text-xs text-muted mt-8">Quote: ${esc(item.quote)}</div>` : '';
      const confidence = item.confidence ? `<div class="text-xs text-muted mt-8">Confidence: ${esc(item.confidence)}</div>` : '';
      html += `<li class="text-sm mb-8">${esc(detail)}${source}${url}${quote}${confidence}</li>`;
    }
    html += '</ul></div>';
  }

  if (Array.isArray(report.sources) && report.sources.length) {
    html += '<div class="mb-8"><strong class="text-xs text-muted">Sources</strong>';
    html += '<ul style="padding-left:16px;margin-top:6px">';
    for (const src of report.sources) {
      if (!src) continue;
      if (typeof src === 'string') {
        html += `<li class="text-sm mb-8">${esc(src)}</li>`;
        continue;
      }
      const title = src.title || src.source || 'source';
      const publisher = src.publisher ? ` (${esc(src.publisher)})` : '';
      const url = src.url ? ` - ${esc(src.url)}` : '';
      html += `<li class="text-sm mb-8">${esc(title)}${publisher}${url}</li>`;
    }
    html += '</ul></div>';
  }

  if (Array.isArray(report.gaps) && report.gaps.length) {
    html += '<div class="mb-8"><strong class="text-xs text-muted">Gaps</strong><ul style="padding-left:16px;margin-top:6px">';
    html += report.gaps.map(g => `<li class="text-sm mb-8">${esc(g)}</li>`).join('');
    html += '</ul></div>';
  }

  if (Array.isArray(report.recommendations) && report.recommendations.length) {
    html += '<div class="mb-8"><strong class="text-xs text-muted">Recommendations</strong><ul style="padding-left:16px;margin-top:6px">';
    html += report.recommendations.map(r => `<li class="text-sm mb-8">${esc(r)}</li>`).join('');
    html += '</ul></div>';
  }

  if (report.confidence !== undefined && report.confidence !== null && report.confidence !== '') {
    html += `<div class="mb-8"><strong class="text-xs text-muted">Confidence</strong>`;
    html += `<div class="text-sm">${esc(String(report.confidence))}</div></div>`;
  }

  // Fallback: if schema differs, still expose all fields.
  if (!html) {
    html = `<div class="json-block">${esc(formatJson(report))}</div>`;
  }
  return html;
}
