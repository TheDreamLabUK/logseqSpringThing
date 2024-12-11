/**
 * Force-directed graph functionality is permanently disabled in favor of server-side calculations.
 * This file remains as a placeholder to maintain compatibility with existing code that may check
 * these values, but the functionality cannot be enabled.
 */

// Force-directed is permanently disabled
(window as any).__FORCE_DIRECTED_CLIENT = false;

// Always returns false since force-directed is permanently disabled
export function isForceDirectedEnabled(): boolean {
  return false;
}

// No-op since force-directed cannot be enabled
export function setForceDirected(_enabled: boolean): void {
  console.warn('Force-directed graph is permanently disabled in favor of server-side calculations.');
}
