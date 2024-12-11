import { VALIDATION } from '../constants/visualization';
import type { ClientDebugSettings } from '../types/websocket';

// Validate a single position value
export function isValidPosition(value: number): boolean {
    return !isNaN(value) && 
           isFinite(value) && 
           value >= VALIDATION.MIN_POSITION && 
           value <= VALIDATION.MAX_POSITION;
}

// Validate a single velocity value
export function isValidVelocity(value: number): boolean {
    return !isNaN(value) && 
           isFinite(value) && 
           value >= VALIDATION.MIN_VELOCITY && 
           value <= VALIDATION.MAX_VELOCITY;
}

// Check if position has changed enough to warrant an update
export function hasSignificantChange(oldPos: number, newPos: number): boolean {
    return Math.abs(oldPos - newPos) > VALIDATION.POSITION_CHANGE_THRESHOLD;
}

// Validate binary data size
export function isValidBinarySize(byteLength: number, nodeCount: number): boolean {
    const expectedSize = 4 + (nodeCount * VALIDATION.EXPECTED_BINARY_SIZE); // 4 bytes header + 24 bytes per node
    return byteLength === expectedSize;
}

// Validate and clamp position values
export function clampPosition(value: number): number {
    if (!isFinite(value) || isNaN(value)) return 0;
    return Math.max(VALIDATION.MIN_POSITION, Math.min(VALIDATION.MAX_POSITION, value));
}

// Validate and clamp velocity values
export function clampVelocity(value: number): number {
    if (!isFinite(value) || isNaN(value)) return 0;
    return Math.max(VALIDATION.MIN_VELOCITY, Math.min(VALIDATION.MAX_VELOCITY, value));
}

// Rate limiting helper
export class UpdateThrottler {
    private lastUpdateTime: number = 0;
    private pendingUpdates: ArrayBuffer[] = [];
    private batchTimeout: number | null = null;

    constructor(private minInterval: number = VALIDATION.UPDATE_INTERVAL) {}

    canUpdate(): boolean {
        const now = performance.now();
        return now - this.lastUpdateTime >= this.minInterval;
    }

    addUpdate(data: ArrayBuffer): void {
        this.pendingUpdates.push(data);
        
        // Start batch timeout if not already started
        if (this.batchTimeout === null && this.pendingUpdates.length < VALIDATION.BATCH_SIZE) {
            this.batchTimeout = window.setTimeout(() => this.processBatch(), this.minInterval);
        } else if (this.pendingUpdates.length >= VALIDATION.BATCH_SIZE) {
            // Process immediately if batch size reached
            if (this.batchTimeout !== null) {
                window.clearTimeout(this.batchTimeout);
                this.batchTimeout = null;
            }
            this.processBatch();
        }
    }

    private processBatch(): void {
        if (this.pendingUpdates.length === 0) return;

        // Use the most recent update
        const latestUpdate = this.pendingUpdates[this.pendingUpdates.length - 1];
        this.pendingUpdates = [];
        this.lastUpdateTime = performance.now();
        this.batchTimeout = null;

        // Notify listeners
        if (this.onUpdate) {
            this.onUpdate(latestUpdate);
        }
    }

    onUpdate: ((data: ArrayBuffer) => void) | null = null;

    reset(): void {
        this.pendingUpdates = [];
        if (this.batchTimeout !== null) {
            window.clearTimeout(this.batchTimeout);
            this.batchTimeout = null;
        }
    }
}

// Enhanced debug logging helper
export class DebugLogger {
    private static instance: DebugLogger;
    private enabled: boolean = false;
    private websocketDebug: boolean = false;
    private dataDebug: boolean = false;
    private binaryHeaders: boolean = false;
    private fullJson: boolean = false;

    private constructor() {}

    static getInstance(): DebugLogger {
        if (!DebugLogger.instance) {
            DebugLogger.instance = new DebugLogger();
        }
        return DebugLogger.instance;
    }

    configure(settings: { client_debug: ClientDebugSettings }): void {
        if (settings?.client_debug) {
            this.enabled = settings.client_debug.enabled;
            this.websocketDebug = settings.client_debug.enable_websocket_debug;
            this.dataDebug = settings.client_debug.enable_data_debug;
            this.binaryHeaders = settings.client_debug.log_binary_headers;
            this.fullJson = settings.client_debug.log_full_json;
            console.info('Debug logger configured:', {
                enabled: this.enabled,
                websocketDebug: this.websocketDebug,
                dataDebug: this.dataDebug,
                binaryHeaders: this.binaryHeaders,
                fullJson: this.fullJson
            });
        }
    }

    log(type: string, message: string, data?: any): void {
        const timestamp = new Date().toISOString();
        const prefix = `[${type.toUpperCase()} ${timestamp}]`;

        switch (type) {
            case 'websocket':
                if (this.enabled || this.websocketDebug) {
                    if (data instanceof ArrayBuffer) {
                        console.debug(`${prefix} ${message}`, `Binary data size: ${data.byteLength} bytes`);
                    } else {
                        console.debug(`${prefix} ${message}`, data);
                    }
                }
                break;
            case 'binary':
                if (this.enabled || this.binaryHeaders) {
                    if (data instanceof ArrayBuffer) {
                        const view = new DataView(data.slice(0, Math.min(16, data.byteLength)));
                        const hexDump = Array.from(new Uint8Array(view.buffer))
                            .map(b => b.toString(16).padStart(2, '0'))
                            .join(' ');
                        console.debug(`${prefix} ${message}`, `Header: ${hexDump}`);
                    } else {
                        console.debug(`${prefix} ${message}`, data);
                    }
                }
                break;
            case 'json':
                if (this.enabled || this.fullJson) {
                    try {
                        const formatted = typeof data === 'object' ? 
                            JSON.stringify(data, null, 2) : data;
                        console.debug(`${prefix} ${message}`, formatted);
                    } catch (error) {
                        console.debug(`${prefix} ${message}`, data);
                    }
                }
                break;
            case 'data':
                if (this.enabled || this.dataDebug) {
                    console.debug(`${prefix} ${message}`, data);
                }
                break;
            default:
                if (this.enabled) {
                    console.debug(`${prefix} ${message}`, data);
                }
        }
    }

    error(message: string, error?: any): void {
        const timestamp = new Date().toISOString();
        console.error(`[ERROR ${timestamp}] ${message}`, error);
    }

    warn(message: string, data?: any): void {
        if (this.enabled) {
            const timestamp = new Date().toISOString();
            console.warn(`[WARN ${timestamp}] ${message}`, data);
        }
    }
}
