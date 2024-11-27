/**
 * Chrome-specific memory performance API
 */
export interface MemoryInfo {
  jsHeapSizeLimit: number;
  totalJSHeapSize: number;
  usedJSHeapSize: number;
}

/**
 * Extended Performance interface with Chrome memory info
 */
export interface ExtendedPerformance extends Performance {
  memory?: MemoryInfo;
}

declare global {
  interface Window {
    performance: ExtendedPerformance;
  }
}
