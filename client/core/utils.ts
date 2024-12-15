/**
 * Core utilities for the LogseqXR visualization system
 */

import { Vector3 } from './types';
import { THROTTLE_INTERVAL } from './constants';

// Debug logging utility
export interface Logger {
  log: (message: string, ...args: any[]) => void;
  error: (message: string, ...args: any[]) => void;
  warn: (message: string, ...args: any[]) => void;
  debug: (message: string, ...args: any[]) => void;
  info: (message: string, ...args: any[]) => void;
}

export function createLogger(namespace: string): Logger {
  return {
    log: (message: string, ...args: any[]) => console.log(`[${namespace}] ${message}`, ...args),
    error: (message: string, ...args: any[]) => console.error(`[${namespace}] ${message}`, ...args),
    warn: (message: string, ...args: any[]) => console.warn(`[${namespace}] ${message}`, ...args),
    debug: (message: string, ...args: any[]) => console.debug(`[${namespace}] ${message}`, ...args),
    info: (message: string, ...args: any[]) => console.info(`[${namespace}] ${message}`, ...args)
  };
}

// Update throttler for performance optimization
export class UpdateThrottler {
  private lastUpdate: number = 0;
  private throttleInterval: number;

  constructor(throttleInterval: number = THROTTLE_INTERVAL) {
    this.throttleInterval = throttleInterval;
  }

  shouldUpdate(): boolean {
    const now = performance.now();
    if (now - this.lastUpdate >= this.throttleInterval) {
      this.lastUpdate = now;
      return true;
    }
    return false;
  }

  reset(): void {
    this.lastUpdate = 0;
  }
}

// Vector operations
export const vectorOps = {
  add: (a: Vector3, b: Vector3): Vector3 => ({
    x: a.x + b.x,
    y: a.y + b.y,
    z: a.z + b.z
  }),

  subtract: (a: Vector3, b: Vector3): Vector3 => ({
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z
  }),

  multiply: (v: Vector3, scalar: number): Vector3 => ({
    x: v.x * scalar,
    y: v.y * scalar,
    z: v.z * scalar
  }),

  divide: (v: Vector3, scalar: number): Vector3 => ({
    x: v.x / scalar,
    y: v.y / scalar,
    z: v.z / scalar
  }),

  length: (v: Vector3): number => 
    Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),

  normalize: (v: Vector3): Vector3 => {
    const len = vectorOps.length(v);
    return len > 0 ? vectorOps.divide(v, len) : { x: 0, y: 0, z: 0 };
  },

  distance: (a: Vector3, b: Vector3): number => 
    vectorOps.length(vectorOps.subtract(a, b)),

  // Convert position array to Vector3
  fromArray: (arr: number[]): Vector3 => ({
    x: arr[0] || 0,
    y: arr[1] || 0,
    z: arr[2] || 0
  })
};

// Scale utilities
export const scaleOps = {
  // Normalize a value between min and max
  normalize: (value: number, min: number, max: number): number => {
    return Math.min(max, Math.max(min, value));
  },

  // Map a value from one range to another
  mapRange: (value: number, inMin: number, inMax: number, outMin: number, outMax: number): number => {
    // First normalize to 0-1
    const normalized = (value - inMin) / (inMax - inMin);
    // Then map to output range
    return outMin + normalized * (outMax - outMin);
  },

  // Scale node size from server range to visualization range
  normalizeNodeSize: (size: number, serverMin: number = 20, serverMax: number = 30, visMin: number = 0.15, visMax: number = 0.4): number => {
    return scaleOps.mapRange(size, serverMin, serverMax, visMin, visMax);
  }
};

// Data validation utilities
export const validateGraphData = (data: any): boolean => {
  if (!data || typeof data !== 'object') return false;
  if (!Array.isArray(data.nodes) || !Array.isArray(data.edges)) return false;
  
  // Validate nodes
  for (const node of data.nodes) {
    if (!node.id) return false;
    // Allow position to be either array or Vector3
    if (node.position) {
      if (Array.isArray(node.position)) {
        if (node.position.length !== 3 || 
            typeof node.position[0] !== 'number' ||
            typeof node.position[1] !== 'number' ||
            typeof node.position[2] !== 'number') {
          return false;
        }
      } else if (typeof node.position === 'object') {
        if (typeof node.position.x !== 'number' ||
            typeof node.position.y !== 'number' ||
            typeof node.position.z !== 'number') {
          return false;
        }
      } else {
        return false;
      }
    }
  }
  
  // Validate edges
  for (const edge of data.edges) {
    if (!edge.source || !edge.target) return false;
  }
  
  return true;
};

// Binary data helpers
export const binaryToFloat32Array = (buffer: ArrayBuffer): Float32Array => {
  return new Float32Array(buffer);
};

export const float32ArrayToPositions = (array: Float32Array): Vector3[] => {
  const positions: Vector3[] = [];
  for (let i = 0; i < array.length; i += 3) {
    positions.push({
      x: array[i],
      y: array[i + 1],
      z: array[i + 2]
    });
  }
  return positions;
};

// Error handling utility
export class VisualizationError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = 'VisualizationError';
  }
}

// Performance monitoring
export class PerformanceMonitor {
  private frames: number = 0;
  private lastTime: number = performance.now();
  private fps: number = 0;

  update(): void {
    this.frames++;
    const now = performance.now();
    const delta = now - this.lastTime;

    if (delta >= 1000) {
      this.fps = (this.frames * 1000) / delta;
      this.frames = 0;
      this.lastTime = now;
    }
  }

  getFPS(): number {
    return Math.round(this.fps);
  }
}
