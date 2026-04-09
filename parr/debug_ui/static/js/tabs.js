/* Tab initialization and state management. */
import { state } from './state.js';

export function initTabs() {
  // Main tabs
  document.querySelectorAll('.tabs').forEach(tabBar => {
    const group = tabBar.dataset.group;
    tabBar.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        state.activeTabs[group] = target;
        // Update buttons
        tabBar.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === target));
        // Update panels
        const container = tabBar.parentElement;
        container.querySelectorAll(`:scope > .tab-panel[data-tab]`).forEach(p => {
          p.classList.toggle('active', p.dataset.tab === target);
        });
      });
    });
    // Restore active tab or default to first
    const saved = state.activeTabs[group];
    const firstBtn = tabBar.querySelector('.tab-btn');
    const activeTab = saved || (firstBtn ? firstBtn.dataset.tab : null);
    if (activeTab) {
      tabBar.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === activeTab));
      const container = tabBar.parentElement;
      container.querySelectorAll(`:scope > .tab-panel[data-tab]`).forEach(p => {
        p.classList.toggle('active', p.dataset.tab === activeTab);
      });
    }
  });

  // Mini tabs (tool I/O)
  document.querySelectorAll('.mini-tabs').forEach(tabBar => {
    const group = tabBar.dataset.group;
    tabBar.querySelectorAll('.mini-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        state.activeTabs[group] = target;
        tabBar.querySelectorAll('.mini-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === target));
        const container = tabBar.parentElement;
        container.querySelectorAll(`:scope > .mini-panel[data-tab]`).forEach(p => {
          p.classList.toggle('active', p.dataset.tab === target);
        });
      });
    });
    const saved = state.activeTabs[group];
    const firstBtn = tabBar.querySelector('.mini-tab');
    const activeTab = saved || (firstBtn ? firstBtn.dataset.tab : null);
    if (activeTab) {
      tabBar.querySelectorAll('.mini-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === activeTab));
      const container = tabBar.parentElement;
      container.querySelectorAll(`:scope > .mini-panel[data-tab]`).forEach(p => {
        p.classList.toggle('active', p.dataset.tab === activeTab);
      });
    }
  });
}
