/* Collapse/expand state persistence for details elements and inline toggles. */
import { state } from './state.js';

export function saveCollapseState() {
  const el = document.getElementById('content');
  const scrollTop = el ? el.scrollTop : 0;
  document.querySelectorAll('#session-detail details[data-cid]').forEach(d => {
    if (d.open) state.openCollapse.add(d.dataset.cid);
    else state.openCollapse.delete(d.dataset.cid);
  });
  document.querySelectorAll('#session-detail .cs-tool[data-cstid]').forEach(d => {
    if (d.classList.contains('open')) state.openInlineTools.add(d.dataset.cstid);
    else state.openInlineTools.delete(d.dataset.cstid);
  });
  document.querySelectorAll('#session-detail .cs-spawn[data-cssid]').forEach(d => {
    if (d.classList.contains('open')) state.openInlineSpawns.add(d.dataset.cssid);
    else state.openInlineSpawns.delete(d.dataset.cssid);
  });
  // Save phase collapse state
  document.querySelectorAll('#session-detail .cs-phase-wrap[data-phid]').forEach(d => {
    if (d.classList.contains('open')) state.openPhases.add(d.dataset.phid);
    else state.openPhases.delete(d.dataset.phid);
  });
  // Save event collapse state (user view v3)
  document.querySelectorAll('#session-detail .uv-evt[data-bid]').forEach(d => {
    if (d.classList.contains('open')) state.openBlocks.add(d.dataset.bid);
    else state.openBlocks.delete(d.dataset.bid);
  });
  return scrollTop;
}

export function restoreCollapseState(scrollTop) {
  document.querySelectorAll('#session-detail details[data-cid]').forEach(d => {
    d.open = state.openCollapse.has(d.dataset.cid);
  });
  document.querySelectorAll('#session-detail .cs-tool[data-cstid]').forEach(d => {
    d.classList.toggle('open', state.openInlineTools.has(d.dataset.cstid));
  });
  document.querySelectorAll('#session-detail .cs-spawn[data-cssid]').forEach(d => {
    d.classList.toggle('open', state.openInlineSpawns.has(d.dataset.cssid));
  });
  // Restore phase collapse state
  document.querySelectorAll('#session-detail .cs-phase-wrap[data-phid]').forEach(d => {
    d.classList.toggle('open', state.openPhases.has(d.dataset.phid));
  });
  // Restore event collapse state (user view v3)
  document.querySelectorAll('#session-detail .uv-evt[data-bid]').forEach(d => {
    d.classList.toggle('open', state.openBlocks.has(d.dataset.bid));
  });
  if (scrollTop !== undefined) {
    const el = document.getElementById('content');
    if (el) el.scrollTop = scrollTop;
  }
}

export function toggleToolRow(rowEl) {
  if (!rowEl || !rowEl.parentElement) return;
  const container = rowEl.parentElement;
  container.classList.toggle('open');
  const id = container.dataset.cstid;
  if (!id) return;
  if (container.classList.contains('open')) state.openInlineTools.add(id);
  else state.openInlineTools.delete(id);
}

export function toggleSpawnRow(rowEl) {
  if (!rowEl || !rowEl.parentElement) return;
  const container = rowEl.parentElement;
  container.classList.toggle('open');
  const id = container.dataset.cssid;
  if (!id) return;
  if (container.classList.contains('open')) state.openInlineSpawns.add(id);
  else state.openInlineSpawns.delete(id);
}

export function togglePhaseSection(rowEl) {
  if (!rowEl || !rowEl.parentElement) return;
  const container = rowEl.parentElement;
  container.classList.toggle('open');
  const id = container.dataset.phid;
  if (!id) return;
  if (container.classList.contains('open')) state.openPhases.add(id);
  else state.openPhases.delete(id);
}

export function toggleBlock(rowEl) {
  if (!rowEl || !rowEl.parentElement) return;
  const container = rowEl.parentElement;
  container.classList.toggle('open');
  const id = container.dataset.bid;
  if (!id) return;
  if (container.classList.contains('open')) state.openBlocks.add(id);
  else state.openBlocks.delete(id);
}
