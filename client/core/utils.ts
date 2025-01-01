/**
 * Core utilities for the LogseqXR visualization system
 */

import { Vector3 } from './types';
import { THROTTLE_INTERVAL } from './constants';

// Debug logging utility
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface Logger {
  debug: (message: string, ...args: unknown[]) => void;
  info: (message: string, ...args: unknown[]) => void;
  warn: (message: string, ...args: unknown[]) => void;
  error: (message: string, ...args: unknown[]) => void;
}

export function createLogger(namespace: string): Logger {
  return {
    debug: (message: string, ...args: unknown[]) => {
      // Use debug logging only in development
      if (process.env.NODE_ENV === 'development') {
        console.debug(`[${namespace}] ${message}`, ...args);
      }
    },
    info: (message: string, ...args: unknown[]) => {
      console.info(`[${namespace}] ${message}`, ...args);
    },
    warn: (message: string, ...args: unknown[]) => {
      console.warn(`[${namespace}] ${message}`, ...args);
    },
    error: (message: string, ...args: unknown[]) => {
      console.error(`[${namespace}] ${message}`, ...args);
    }
  };
}

// Case conversion utilities
export const camelToSnakeCase = (str: string): string => {
  return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
};

export const snakeToCamelCase = (str: string): string => {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
};

export const convertObjectKeysToSnakeCase = (obj: Record<string, unknown>): Record<string, unknown> => {
  if (Array.isArray(obj)) {
    return obj.map(item => convertObjectKeysToSnakeCase(item));
  }
  
  if (obj !== null && typeof obj === 'object') {
    return Object.keys(obj).reduce((acc, key) => {
      const snakeKey = camelToSnakeCase(key);
      acc[snakeKey] = convertObjectKeysToSnakeCase(obj[key]);
      return acc;
    }, {} as Record<string, unknown>);
  }
  
  return obj;
};

export const convertObjectKeysToCamelCase = (obj: Record<string, unknown>): Record<string, unknown> => {
  if (Array.isArray(obj)) {
    return obj.map(item => convertObjectKeysToCamelCase(item));
  }
  
  if (obj !== null && typeof obj === 'object') {
    return Object.keys(obj).reduce((acc, key) => {
      const camelKey = snakeToCamelCase(key);
      acc[camelKey] = convertObjectKeysToCamelCase(obj[key]);
      return acc;
    }, {} as Record<string, unknown>);
  }
  
  return obj;
};

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
export const validateGraphData = (data: Record<string, unknown>): boolean => {
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

// Type definitions for utility functions
export type LogFunction = (message: string, ...args: unknown[]) => void;
export type ErrorCallback = (error: Error) => void;
export type SuccessCallback<T> = (result: T) => void;

// Constants
const DEFAULT_TIMEOUT = 5000;

// Logging utilities
export function createLogger(namespace: string): Logger {
    return {
        log: (message: string, ...args: unknown[]) => {
            console.log(`[${namespace}] ${message}`, ...args);
        },
        error: (message: string, ...args: unknown[]) => {
            console.error(`[${namespace}] ${message}`, ...args);
        },
        warn: (message: string, ...args: unknown[]) => {
            console.warn(`[${namespace}] ${message}`, ...args);
        },
        info: (message: string, ...args: unknown[]) => {
            console.info(`[${namespace}] ${message}`, ...args);
        }
    };
}

// Async utilities
export function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export async function withTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number = DEFAULT_TIMEOUT
): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Operation timed out')), timeoutMs);
    });
    return Promise.race([promise, timeoutPromise]);
}

export async function withRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    delayMs: number = 1000
): Promise<T> {
    let lastError: Error | undefined;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));
            if (attempt < maxRetries - 1) {
                await delay(delayMs * Math.pow(2, attempt));
            }
        }
    }
    
    throw lastError || new Error('Operation failed after retries');
}

// Type utilities
export type JsonValue = string | number | boolean | null | JsonObject | JsonArray;
export type JsonObject = { [key: string]: JsonValue };
export type JsonArray = JsonValue[];

export type EventCallback<T> = (data: T) => void;
export type ErrorCallback = (error: Error) => void;
export type SuccessCallback<T> = (result: T) => void;

// Event handling utilities
export function createEventEmitter<T>() {
    const listeners = new Set<EventCallback<T>>();
    
    return {
        on(callback: EventCallback<T>) {
            listeners.add(callback);
            return () => listeners.delete(callback);
        },
        emit(data: T) {
            listeners.forEach(listener => listener(data));
        },
        clear() {
            listeners.clear();
        }
    };
}

// Object utilities
export function deepMerge<T extends Record<string, JsonValue>>(
    target: T,
    source: DeepPartial<T>
): T {
    const result = { ...target };
    
    for (const key in source) {
        const value = source[key];
        if (value && typeof value === 'object' && !Array.isArray(value)) {
            result[key] = deepMerge(
                (target[key] as Record<string, JsonValue>) || {},
                value as DeepPartial<Record<string, JsonValue>>
            ) as T[typeof key];
        } else {
            result[key] = value as T[typeof key];
        }
    }
    
    return result;
}

// Validation utilities
export function isNonNullable<T>(value: T): value is NonNullable<T> {
    return value !== null && value !== undefined;
}

export function assertNonNullable<T>(
    value: T,
    message = 'Value must not be null or undefined'
): asserts value is NonNullable<T> {
    if (!isNonNullable(value)) {
        throw new Error(message);
    }
}

// Math utilities
export function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

export function lerp(start: number, end: number, t: number): number {
    return start + (end - start) * clamp(t, 0, 1);
}

// String utilities
export function formatBytes(bytes: number, decimals = 2): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

// DOM utilities
export function createElement<K extends keyof HTMLElementTagNameMap>(
    tagName: K,
    options?: ElementCreationOptions
): HTMLElementTagNameMap[K] {
    return document.createElement(tagName, options);
}

export function removeElement(element: Element): void {
    element.parentElement?.removeChild(element);
}

// URL utilities
export function isValidUrl(url: string): boolean {
    try {
        new URL(url);
        return true;
    } catch {
        return false;
    }
}

export function joinPaths(...paths: string[]): string {
    return paths
        .map(path => path.replace(/^\/+|\/+$/g, ''))
        .filter(Boolean)
        .join('/');
}

// Debounce utility
export function debounce<T extends (...args: unknown[]) => void>(
    func: T,
    wait: number
): (...args: Parameters<T>) => void {
    let timeout: ReturnType<typeof setTimeout> | undefined;

    return (...args: Parameters<T>) => {
        const later = () => {
            timeout = undefined;
            func(...args);
        };

        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle utility
export function throttle<T extends (...args: unknown[]) => void>(
    func: T,
    limit: number
): (...args: Parameters<T>) => void {
    let inThrottle = false;
    let lastFunc: ReturnType<typeof setTimeout>;
    let lastRan: number;

    return (...args: Parameters<T>) => {
        if (!inThrottle) {
            func(...args);
            lastRan = Date.now();
            inThrottle = true;
        } else {
            clearTimeout(lastFunc);
            lastFunc = setTimeout(() => {
                if (Date.now() - lastRan >= limit) {
                    func(...args);
                    lastRan = Date.now();
                }
            }, limit - (Date.now() - lastRan));
        }
    };
}

// Color utilities
export interface RGB {
    r: number;
    g: number;
    b: number;
}

export interface HSL {
    h: number;
    s: number;
    l: number;
}

export function rgbToHsl(rgb: RGB): HSL {
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0;
    let s = 0;
    const l = (max + min) / 2;

    if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

        switch (max) {
            case r:
                h = (g - b) / d + (g < b ? 6 : 0);
                break;
            case g:
                h = (b - r) / d + 2;
                break;
            case b:
                h = (r - g) / d + 4;
                break;
        }

        h /= 6;
    }

    return { h, s, l };
}

export function hslToRgb(hsl: HSL): RGB {
    let r = 0;
    let g = 0;
    let b = 0;

    if (hsl.s === 0) {
        r = g = b = hsl.l;
    } else {
        const hue2rgb = (p: number, q: number, t: number): number => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        const q = hsl.l < 0.5 ? hsl.l * (1 + hsl.s) : hsl.l + hsl.s - hsl.l * hsl.s;
        const p = 2 * hsl.l - q;

        r = hue2rgb(p, q, hsl.h + 1 / 3);
        g = hue2rgb(p, q, hsl.h);
        b = hue2rgb(p, q, hsl.h - 1 / 3);
    }

    return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
    };
}

export function hexToRgb(hex: string): RGB | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
        ? {
              r: parseInt(result[1], 16),
              g: parseInt(result[2], 16),
              b: parseInt(result[3], 16)
          }
        : null;
}

export function rgbToHex(rgb: RGB): string {
    const toHex = (n: number): string => {
        const hex = n.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    };

    return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`;
}

// Environment utilities
export function getEnvironmentVariable(key: string): string | undefined {
    if (typeof process !== 'undefined' && process.env && process.env[key]) {
        return process.env[key];
    }
    return undefined;
}

export function isProduction(): boolean {
    return process.env.NODE_ENV === 'production';
}

export function isDevelopment(): boolean {
    return process.env.NODE_ENV === 'development';
}

export function isTest(): boolean {
    return process.env.NODE_ENV === 'test';
}

// Sleep utility
export function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Deferred promise utility
export function createDeferredPromise<T>(): {
    promise: Promise<T>;
    resolve: (value: T | PromiseLike<T>) => void;
    reject: (reason?: unknown) => void;
} {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });

    return { promise, resolve, reject };
}

// Format bytes utility
export function formatBytes(bytes: number, decimals = 2): string {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

// Deep clone utility
export function deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item)) as unknown as T;
    }

    return Object.fromEntries(
        Object.entries(obj as Record<string, unknown>).map(([key, value]) => [
            key,
            deepClone(value)
        ])
    ) as T;
}

// Deep equal utility
export function deepEqual(a: unknown, b: unknown): boolean {
    if (a === b) return true;

    if (a === null || b === null) return false;
    if (typeof a !== 'object' || typeof b !== 'object') return false;

    const keysA = Object.keys(a as object);
    const keysB = Object.keys(b as object);

    if (keysA.length !== keysB.length) return false;

    for (const key of keysA) {
        if (!keysB.includes(key)) return false;
        if (!deepEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key])) return false;
    }

    return true;
}

// Memoize utility
export function memoize<T extends (...args: unknown[]) => unknown>(
    fn: T,
    keyFn?: (...args: Parameters<T>) => string
): T {
    const cache = new Map<string, ReturnType<T>>();

    return ((...args: Parameters<T>) => {
        const key = keyFn ? keyFn(...args) : JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn(...args);
        cache.set(key, result as ReturnType<T>);
        return result;
    }) as T;
}

// Retry utility
export function retry<T>(
    fn: () => Promise<T>,
    options: {
        maxAttempts?: number;
        delay?: number;
        backoff?: number;
        onRetry?: (attempt: number, error: Error) => void;
    } = {}
): Promise<T> {
    const {
        maxAttempts = 3,
        delay = 1000,
        backoff = 2,
        onRetry = () => {}
    } = options;

    return new Promise((resolve, reject) => {
        let attempts = 0;

        const attempt = async () => {
            try {
                const result = await fn();
                resolve(result);
            } catch (error) {
                attempts++;
                if (attempts >= maxAttempts) {
                    reject(error);
                    return;
                }

                onRetry(attempts, error as Error);
                const nextDelay = delay * Math.pow(backoff, attempts - 1);
                setTimeout(attempt, nextDelay);
            }
        };

        attempt();
    });
}
