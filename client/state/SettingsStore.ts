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
    private initializationPromise: Promise<void> | null = null;
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
        // If already initialized or initializing, return existing promise
        if (this.initialized) {
            return Promise.resolve();
        }
        if (this.initializationPromise) {
            return this.initializationPromise;
        }

        this.initializationPromise = (async () => {
            try {
                // Try to fetch settings from API
                try {
                    const response = await fetch(API_ENDPOINTS.SETTINGS);
                    if (response.ok) {
                        const settings = await response.json();
                        this.settings = { ...defaultSettings, ...settings };
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
                // Use defaults on error
                this.settings = { ...defaultSettings };
                this.initialized = true;
            }
        })();

        return this.initializationPromise;
    }

    public isInitialized(): boolean {
        return this.initialized;
    }

    public async subscribe(path: string, callback: SettingsChangeCallback): Promise<() => void> {
        // Wait for initialization if not already initialized
        if (!this.initialized) {
            await this.initialize();
        }

        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path)?.add(callback);

        // Immediately call callback with current value
        const value = this.get(path);
        if (value !== undefined) {
            callback(path, value);
        }

        // Return unsubscribe function
        return () => {
            const pathSubscribers = this.subscribers.get(path);
            if (pathSubscribers) {
                pathSubscribers.delete(callback);
                if (pathSubscribers.size === 0) {
                    this.subscribers.delete(path);
                }
            }
        };
    }

    private startSyncTimer(): void {
        if (this.syncTimer !== null) {
            window.clearInterval(this.syncTimer);
        }
        this.syncTimer = window.setInterval(
            () => this.saveChanges(),
            this.options.syncInterval
        ) as unknown as number;
    }

    private async saveChanges(): Promise<void> {
        if (this.pendingChanges.size === 0) {
            return;
        }

        const changes = Array.from(this.pendingChanges);
        this.pendingChanges.clear();

        for (const path of changes) {
            try {
                const value = this.get(path);
                await this.saveSetting(path, value);
            } catch (error) {
                logger.error(`Failed to save setting ${path}:`, error);
                // Re-add to pending changes for retry
                this.pendingChanges.add(path);
            }
        }
    }

    private async saveSetting(path: string, value: unknown): Promise<void> {
        const response = await fetch(API_ENDPOINTS.SETTINGS_UPDATE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, value })
        });

        if (!response.ok) {
            throw new Error(`Failed to save setting ${path}: ${response.statusText}`);
        }
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
