import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { defaultSettings } from './defaultSettings';
import { API_ENDPOINTS } from '../core/constants';

const logger = createLogger('SettingsStore');

export interface SettingsStoreOptions {
    autoSave?: boolean;
    syncInterval?: number;
}

const defaultSettingsStoreOptions: Required<SettingsStoreOptions> = {
    autoSave: true,
    syncInterval: 5000
};

export type SettingsChangeCallback = (path: string, value: unknown) => void;

export class SettingsStore {
    private static instance: SettingsStore | null = null;
    private settings: Settings;
    private initialized: boolean = false;
    private pendingChanges: Set<string> = new Set();
    private subscribers: Map<string, Set<SettingsChangeCallback>> = new Map();
    private syncTimer: number | null = null;

    private constructor(
        private readonly options: SettingsStoreOptions = defaultSettingsStoreOptions
    ) {
        // Initialize with default settings
        this.settings = { ...defaultSettings };
    }

    public static getInstance(options?: SettingsStoreOptions): SettingsStore {
        if (!SettingsStore.instance) {
            SettingsStore.instance = new SettingsStore(options);
        }
        return SettingsStore.instance;
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            // Try to fetch settings from API
            try {
                const response = await fetch(API_ENDPOINTS.SETTINGS);
                if (response.ok) {
                    const flatSettings = await response.json();
                    this.settings = this.unflattenSettings(flatSettings);
                    logger.info('Settings loaded from API');
                } else {
                    throw new Error(`Failed to fetch settings: ${response.statusText}`);
                }
            } catch (error) {
                logger.warn('Failed to fetch settings from API, using defaults');
                this.settings = { ...defaultSettings };
            }

            // Start sync timer if auto-save is enabled
            if (this.options.autoSave) {
                this.startSyncTimer();
            }

            this.initialized = true;
            logger.info('SettingsStore initialized');
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            throw error;
        }
    }

    private startSyncTimer(): void {
        this.syncTimer = window.setInterval(
            () => this.saveChanges(),
            this.options.syncInterval
        ) as unknown as number;
    }

    private async saveChanges(): Promise<void> {
        if (this.pendingChanges.size === 0) {
            return;
        }

        for (const path of this.pendingChanges) {
            const [category, setting] = path.split('.');
            if (!category || !setting) continue;

            try {
                const value = this.get(path);
                const response = await fetch(API_ENDPOINTS.SETTINGS_ITEM(category, setting), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ value })
                });

                if (!response.ok) {
                    throw new Error(`Failed to save setting ${path}: ${response.statusText}`);
                }
            } catch (error) {
                logger.error(`Failed to save setting ${path}:`, error);
            }
        }

        this.pendingChanges.clear();
    }

    public get(path: string): unknown {
        if (!this.initialized) {
            logger.warn('Attempting to access settings before initialization');
            return undefined;
        }
        
        if (!path) {
            return this.settings;
        }
        
        try {
            return path.split('.').reduce((obj: any, key) => {
                if (obj === null || obj === undefined) {
                    throw new Error(`Invalid path: ${path}`);
                }
                return obj[key];
            }, this.settings);
        } catch (error) {
            logger.error(`Error accessing setting at path ${path}:`, error);
            return undefined;
        }
    }

    public set(path: string, value: unknown): void {
        if (!this.initialized) {
            logger.error('Attempting to set settings before initialization');
            throw new Error('SettingsStore not initialized');
        }

        try {
            const parts = path.split('.');
            const lastKey = parts.pop()!;
            const target = parts.reduce((obj: any, key) => {
                if (!(key in obj)) {
                    obj[key] = {};
                }
                return obj[key];
            }, this.settings);

            if (!target || typeof target !== 'object') {
                throw new Error(`Invalid settings path: ${path}`);
            }

            target[lastKey] = value;
            this.pendingChanges.add(path);
            this.notifySubscribers(path, value);
        } catch (error) {
            logger.error(`Error setting value at path ${path}:`, error);
            throw error;
        }
    }

    private unflattenSettings(flatSettings: Record<string, unknown>): Settings {
        const result: any = {};
        
        for (const [path, value] of Object.entries(flatSettings)) {
            const parts = path.split('.');
            let current = result;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) {
                    current[part] = {};
                }
                current = current[part];
            }
            
            current[parts[parts.length - 1]] = value;
        }
        
        return result as Settings;
    }

    public subscribe(path: string, callback: SettingsChangeCallback): () => void {
        if (!this.initialized) {
            logger.warn('Attempting to subscribe before initialization');
            throw new Error('SettingsStore not initialized');
        }

        try {
            if (!this.subscribers.has(path)) {
                this.subscribers.set(path, new Set());
            }
            const pathSubscribers = this.subscribers.get(path)!;
            pathSubscribers.add(callback);

            // Immediately notify subscriber with current value
            const currentValue = this.get(path);
            if (currentValue !== undefined) {
                try {
                    callback(path, currentValue);
                } catch (error) {
                    logger.error(`Error in initial callback for ${path}:`, error);
                }
            }

            return () => {
                const pathSubscribers = this.subscribers.get(path);
                if (pathSubscribers) {
                    pathSubscribers.delete(callback);
                    if (pathSubscribers.size === 0) {
                        this.subscribers.delete(path);
                    }
                }
            };
        } catch (error) {
            logger.error(`Error setting up subscription for ${path}:`, error);
            throw error;
        }
    }

    private notifySubscribers(path: string, value: unknown): void {
        const subscribers = this.subscribers.get(path);
        if (subscribers) {
            subscribers.forEach(callback => {
                try {
                    callback(path, value);
                } catch (error) {
                    logger.error(`Error in settings subscriber for ${path}:`, error);
                }
            });
        }
    }

    public dispose(): void {
        if (this.syncTimer !== null) {
            window.clearInterval(this.syncTimer);
            this.syncTimer = null;
        }
        this.subscribers.clear();
        this.pendingChanges.clear();
        this.initialized = false;
    }
}
