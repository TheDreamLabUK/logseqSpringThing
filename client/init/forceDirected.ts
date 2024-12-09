/**
 * Force-directed graph initialization
 * 
 * This script ensures the client-side force-directed graph is disabled by default
 * to prevent CPU/memory issues. It can be enabled via the settings panel if needed.
 * 
 * Location: client/init/forceDirected.ts
 */

import { VISUALIZATION_CONSTANTS } from '../constants/visualization';

// Override the force-directed setting to ensure it starts disabled
(window as any).__FORCE_DIRECTED_CLIENT = false;

// Export a function to check if force-directed is enabled
export function isForceDirectedEnabled(): boolean {
    return (window as any).__FORCE_DIRECTED_CLIENT && VISUALIZATION_CONSTANTS.FORCE_DIRECTED_CLIENT;
}

// Export a function to toggle force-directed state
export function setForceDirected(enabled: boolean): void {
    (window as any).__FORCE_DIRECTED_CLIENT = enabled;
}

// Initialize with force-directed disabled
setForceDirected(false);
