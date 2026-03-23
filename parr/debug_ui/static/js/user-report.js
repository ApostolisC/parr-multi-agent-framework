/* Polished report renderer for user view — clean layout, inline metadata. */
import { esc, formatJson } from './utils.js';
import { renderMarkdown } from './renderers.js';

/**
 * Render a structured report in a clean, user-friendly format.
 * The answer is the primary output. Supporting sections (evidence,
 * recommendations) show inline metadata — no expandable pills.
 */
export function renderUserReport(report) {
  if (!report || typeof report !== 'object') {
    const text = typeof report === 'string' ? report : formatJson(report);
    return `<div class="ur-section"><div class="ur-prose">${renderMarkdown(text)}</div></div>`;
  }

  let html = '<div class="ur-report">';

  // --- Answer (the primary deliverable) ---
  const answer = report.answer || report.direct_answer;
  if (answer) {
    html += `<div class="ur-answer">${renderMarkdown(String(answer))}</div>`;
  }

  // --- Summary (only if meaningfully different from answer) ---
  if (report.summary && report.summary !== answer) {
    // Skip summary when the answer already exists and is substantive,
    // or when the summary starts with the same words as the answer.
    const answerText = (answer || '').trim();
    const summaryText = report.summary.trim();
    const sameStart = answerText.substring(0, 35).toLowerCase() === summaryText.substring(0, 35).toLowerCase();
    const answerIsSubstantive = answerText.length > 200;
    if (!sameStart && !answerIsSubstantive) {
      html += `<div class="ur-summary">${renderMarkdown(String(report.summary))}</div>`;
    }
  }

  // --- Evidence (inline source/confidence — no expandable pills) ---
  // Only show if no answer exists (evidence as primary), or if evidence
  // contains items with real external sources (URLs, papers, not just model_knowledge)
  const hasRealSources = _hasExternalEvidence(report.evidence);
  if (Array.isArray(report.evidence) && report.evidence.length && (!answer || hasRealSources)) {
    html += '<div class="ur-section">';
    html += `<div class="ur-section-title">Evidence</div>`;
    html += '<ul class="ur-list">';
    for (const item of report.evidence) {
      if (!item) continue;
      if (typeof item === 'string') {
        html += `<li>${esc(item)}</li>`;
      } else {
        const detail = item.detail || item.content || item.finding || formatJson(item);
        html += `<li>${esc(detail)}`;
        // Inline metadata — small text, no expandable pills
        const metaParts = [];
        if (item.source && !_isModelKnowledge(item.source)) metaParts.push(item.source);
        if (item.url) metaParts.push(item.url);
        if (item.confidence && item.confidence !== 'high') metaParts.push(`confidence: ${item.confidence}`);
        if (metaParts.length) {
          html += ` <span class="ur-inline-meta">${esc(metaParts.join(' · '))}</span>`;
        }
        html += `</li>`;
      }
    }
    html += '</ul></div>';
  }

  // --- Key Findings (only show when no answer — as fallback primary content) ---
  if (!answer && Array.isArray(report.key_findings) && report.key_findings.length) {
    html += '<div class="ur-findings">';
    html += `<div class="ur-section-title">Key Findings</div>`;
    for (let i = 0; i < report.key_findings.length; i++) {
      const item = report.key_findings[i];
      if (!item) continue;
      html += _renderFindingCard(item);
    }
    html += '</div>';
  }

  // --- Gaps ---
  if (Array.isArray(report.gaps) && report.gaps.length) {
    html += '<div class="ur-section">';
    html += `<div class="ur-section-title">Gaps</div>`;
    html += '<ul class="ur-list">';
    for (const g of report.gaps) {
      if (g) html += `<li>${esc(g)}</li>`;
    }
    html += '</ul></div>';
  }

  // --- Recommendations ---
  if (Array.isArray(report.recommendations) && report.recommendations.length) {
    html += '<div class="ur-section">';
    html += `<div class="ur-section-title">Recommendations</div>`;
    html += '<ul class="ur-list">';
    for (const r of report.recommendations) {
      if (r) html += `<li>${esc(r)}</li>`;
    }
    html += '</ul></div>';
  }

  // --- Sources (compact, only external) ---
  if (Array.isArray(report.sources) && report.sources.length) {
    const externalSources = report.sources.filter(s => {
      if (!s) return false;
      const text = typeof s === 'string' ? s : (s.title || s.source || '');
      return !_isModelKnowledge(text);
    });
    if (externalSources.length) {
      html += '<div class="ur-section ur-sources">';
      html += `<div class="ur-section-title">Sources</div>`;
      for (const src of externalSources) {
        if (typeof src === 'string') {
          html += `<div class="ur-source-item">${esc(src)}</div>`;
        } else {
          const title = src.title || src.source || 'Source';
          const publisher = src.publisher ? ` — ${esc(src.publisher)}` : '';
          const url = src.url ? ` <span class="ur-source-url">${esc(src.url)}</span>` : '';
          html += `<div class="ur-source-item">${esc(title)}${publisher}${url}</div>`;
        }
      }
      html += '</div>';
    }
  }

  // Fallback: if nothing was rendered, show formatted output
  if (html === '<div class="ur-report">') {
    html += `<div class="ur-prose">${renderMarkdown(formatJson(report))}</div>`;
  }

  html += '</div>';
  return html;
}

// --- Helpers ---

function _isModelKnowledge(source) {
  if (!source) return false;
  const s = source.toLowerCase();
  return s.includes('model_knowledge') || s.includes('model knowledge')
    || s === 'general knowledge' || s === 'internal';
}

function _hasExternalEvidence(evidence) {
  if (!Array.isArray(evidence)) return false;
  return evidence.some(item => {
    if (!item || typeof item === 'string') return false;
    if (item.url) return true;
    if (item.source && !_isModelKnowledge(item.source)) return true;
    return false;
  });
}

function _renderFindingCard(item) {
  const finding = item.finding || item.detail || item.content || formatJson(item);
  let html = `<div class="ur-finding-card"><div class="ur-finding-text">${esc(finding)}</div>`;
  const metaParts = [];
  if (item.source && !_isModelKnowledge(item.source)) metaParts.push(item.source);
  if (item.confidence) metaParts.push(`confidence: ${item.confidence}`);
  if (metaParts.length) {
    html += ` <span class="ur-inline-meta">${esc(metaParts.join(' · '))}</span>`;
  }
  html += '</div>';
  return html;
}
