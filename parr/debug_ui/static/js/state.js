/* Centralized UI state — shared mutable object. */
export const state = {
  currentSessionId: null,
  pollTimer: null,
  sessionPollTimer: null,
  config: { can_start: false, available_roles: [], sse_available: false, can_continue: false },
  sseConnected: false,
  openCollapse: new Set(),
  openInlineTools: new Set(),
  openInlineSpawns: new Set(),
  openPhases: new Set(),
  openBlocks: new Set(),
  lastSessionListHash: '',
  lastDetailHash: '',
  autoSelectNewest: false,
  activeTabs: {},
  searchQuery: '',
  viewMode: 'user',           // 'user' | 'debug'
  lastSessionData: null,       // cached for view-mode re-render
  chatChains: {},              // newSessionId -> parentSessionId
  pendingContinuation: null,   // parentSessionId while waiting for new session
  _durationTimer: null,        // interval ID for live duration updates
};
