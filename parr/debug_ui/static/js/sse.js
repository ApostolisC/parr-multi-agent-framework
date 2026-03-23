/* SSE client for real-time event notifications. */
import { state } from './state.js';

let eventSource = null;

/**
 * Connect to the SSE endpoint.
 * @param {Object} callbacks
 * @param {Function} callbacks.onSessionList  — called when session list may have changed
 * @param {Function} callbacks.onSessionUpdate — called when a session's detail changed
 */
export function connectSSE({ onSessionList, onSessionUpdate }) {
  if (eventSource) eventSource.close();

  eventSource = new EventSource('/api/events');

  eventSource.addEventListener('connected', () => {
    state.sseConnected = true;
  });

  eventSource.addEventListener('session_list', (e) => {
    if (onSessionList) onSessionList(JSON.parse(e.data));
  });

  eventSource.addEventListener('session_update', (e) => {
    if (onSessionUpdate) onSessionUpdate(JSON.parse(e.data));
  });

  eventSource.onopen = () => { state.sseConnected = true; };

  eventSource.onerror = () => {
    state.sseConnected = false;
    // EventSource auto-reconnects with browser-managed backoff
  };
}

export function disconnectSSE() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  state.sseConnected = false;
}
