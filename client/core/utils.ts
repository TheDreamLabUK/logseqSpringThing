/**
 * Core utilities for the LogseqXR visualization system
 */

import { Vector3, Euler, Quaternion, Matrix4 } from 'three';
import { Logger, LogLevel } from './types';
import { THROTTLE_INTERVAL } from './constants';

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
    },
    log: (message: string, ...args: unknown[]) => {
      console.log(`[${namespace}] ${message}`, ...args);
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

export function convertObjectKeysToSnakeCase<T>(obj: T): T extends Array<any> ? Array<Record<string, unknown>> : Record<string, unknown> {
  if (obj === null || typeof obj !== 'object') {
    return obj as any;
  }
  
  if (Array.isArray(obj)) {
    return obj.map(item => convertObjectKeysToSnakeCase(item)) as any;
  }
  
  return Object.keys(obj as object).reduce((acc, key) => {
    const snakeKey = camelToSnakeCase(key);
    acc[snakeKey] = convertObjectKeysToSnakeCase((obj as any)[key]);
    return acc;
  }, {} as Record<string, unknown>) as any;
}

export function convertObjectKeysToCamelCase<T>(obj: T): T extends Array<any> ? Array<Record<string, unknown>> : Record<string, unknown> {
  if (obj === null || typeof obj !== 'object') {
    return obj as any;
  }
  
  if (Array.isArray(obj)) {
    return obj.map(item => convertObjectKeysToCamelCase(item)) as any;
  }
  
  return Object.keys(obj as object).reduce((acc, key) => {
    const camelKey = snakeToCamelCase(key);
    acc[camelKey] = convertObjectKeysToCamelCase((obj as any)[key]);
    return acc;
  }, {} as Record<string, unknown>) as any;
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
  add: (a: Vector3, b: Vector3): Vector3 => {
    return new Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
  },
  subtract: (a: Vector3, b: Vector3): Vector3 => {
    return new Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
  },
  multiply: (v: Vector3, scalar: number): Vector3 => {
    return new Vector3(v.x * scalar, v.y * scalar, v.z * scalar);
  },
  divide: (v: Vector3, scalar: number): Vector3 => {
    return new Vector3(v.x / scalar, v.y / scalar, v.z / scalar);
  },
  length: (v: Vector3): number => 
    Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),
  normalize: (v: Vector3): Vector3 => {
    const len = vectorOps.length(v);
    return len > 0 ? vectorOps.divide(v, len) : new Vector3(0, 0, 0);
  },
  distance: (a: Vector3, b: Vector3): number => 
    vectorOps.length(vectorOps.subtract(a, b)),
  fromArray: (arr: number[]): Vector3 => {
    return new Vector3(arr[0] || 0, arr[1] || 0, arr[2] || 0);
  }
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
    positions.push(new Vector3(array[i], array[i + 1], array[i + 2]));
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

// Constants
const DEFAULT_TIMEOUT = 5000;

// Logging utilities
export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export function deepMerge<T extends Record<string, unknown>>(target: T, source: DeepPartial<T>): T {
    const output = { ...target };

    for (const key in source) {
        if (source.hasOwnProperty(key)) {
            const sourceValue = source[key];
            if (sourceValue && typeof sourceValue === 'object' && !Array.isArray(sourceValue)) {
                if (!(key in target)) {
                    Object.assign(output, { [key]: sourceValue });
                } else {
                    const targetValue = target[key] as Record<string, unknown>;
                    output[key] = deepMerge(targetValue, sourceValue as DeepPartial<typeof targetValue>) as T[typeof key];
                }
            } else {
                Object.assign(output, { [key]: sourceValue });
            }
        }
    }

    return output;
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
export function isObject(item: unknown): item is Record<string, unknown> {
    return item !== null && typeof item === 'object' && !Array.isArray(item);
}

export function generateUUID(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

export function createVector3FromObject(obj: { x: number; y: number; z: number }): Vector3 {
    return new Vector3(obj.x, obj.y, obj.z);
}

export function createEulerFromObject(obj: { x: number; y: number; z: number; order?: string }): Euler {
    return new Euler(obj.x, obj.y, obj.z, obj.order as "XYZ" | "YXZ" | "ZXY" | "ZYX" | "YZX" | "XZY" | undefined);
}

export function createQuaternionFromObject(obj: { x: number; y: number; z: number; w: number }): Quaternion {
    return new Quaternion(obj.x, obj.y, obj.z, obj.w);
}

export function createMatrix4FromArray(array: number[]): Matrix4 {
    return new Matrix4().fromArray(array);
}

export function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export async function retry<T>(
    fn: () => Promise<T>,
    retries: number = 3,
    delay: number = 1000,
    onError?: (error: Error) => void
): Promise<T> {
    let lastError: Error;
    try {
        return await fn();
    } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (retries === 0) {
            throw lastError;
        }
        if (onError) {
            onError(lastError);
        }
        await sleep(delay);
        return retry(fn, retries - 1, delay, onError);
    }
}

export function memoize<T extends (...args: any[]) => any>(
    fn: T,
    resolver?: (...args: Parameters<T>) => string
): T {
    const cache = new Map<string, ReturnType<T>>();

    return function memoized(this: unknown, ...args: Parameters<T>): ReturnType<T> {
        const key = resolver ? resolver(...args) : JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key)!;
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    } as T;
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

// String utilities
export function formatBytes(bytes: number, decimals = 2): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
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
