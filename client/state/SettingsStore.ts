import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { defaultSettings } from './defaultSettings';

const logger = createLogger('SettingsStore');

export type SettingsChangeCallback = (path: string, value: unknown) => void;

export class SettingsStore {
    private static instance: SettingsStore | null = null;
    private settings: Settings;
    private initialized: boolean = false;
    private initializationPromise: Promise<void> | null = null;
    private pendingChanges: Set<string> = new Set();
    private subscribers: Map<string, Set<SettingsChangeCallback>> = new Map();
    private syncTimer: number | null = null;

    private constructor() {
        // Initialize with default settings
        this.settings = { ...defaultSettings };
    }

    public static getInstance(): SettingsStore {
        if (!SettingsStore.instance) {
            SettingsStore.instance = new SettingsStore();
        }
        return SettingsStore.instance;
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return Promise.resolve();
        }
        if (this.initializationPromise) {
            return this.initializationPromise;
        }

        this.initializationPromise = (async () => {
            try {
                // Start with default settings
                this.settings = { ...defaultSettings };

                // Fetch settings from server
                const response = await fetch('/api/settings');
                if (!response.ok) {
                    throw new Error(`Failed to fetch settings: ${response.statusText}`);
                }
                const serverSettings = await response.json();
                
                // Deep merge server settings with defaults
                this.settings = this.deepMerge(this.settings, serverSettings);
                
                // Enable physics by default for graph visualization
                if (this.settings.visualization?.physics) {
                    this.settings.visualization.physics.enabled = true;
                }

                this.initialized = true;
                logger.info('SettingsStore initialized with server settings');
                
                // Start periodic sync
                this.startSync();
            } catch (error) {
                logger.error('Failed to initialize settings:', error);
                // Use defaults on error but still enable physics
                this.settings = { ...defaultSettings };
                if (this.settings.visualization?.physics) {
                    this.settings.visualization.physics.enabled = true;
                }
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

    private deepMerge(target: any, source: any): any {
        const result = { ...target };
        
        for (const key in source) {
            if (source[key] instanceof Object && key in target) {
                result[key] = this.deepMerge(target[key], source[key]);
            } else {
                result[key] = source[key];
            }
        }
        
        return result;
    }

    private startSync(): void {
        // Clear any existing sync timer
        if (this.syncTimer !== null) {
            window.clearInterval(this.syncTimer);
        }

        // Start periodic sync every 30 seconds
        this.syncTimer = window.setInterval(async () => {
            try {
                // Only sync if there are pending changes
                if (this.pendingChanges.size > 0) {
                    const changedSettings: Record<string, unknown> = {};
                    this.pendingChanges.forEach(path => {
                        changedSettings[path] = this.get(path);
                    });

                    const response = await fetch('/api/settings', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(changedSettings),
                    });

                    if (!response.ok) {
                        throw new Error(`Failed to sync settings: ${response.statusText}`);
                    }

                    // Clear pending changes after successful sync
                    this.pendingChanges.clear();
                    logger.debug('Settings synced with server');
                }
            } catch (error) {
                logger.error('Failed to sync settings:', error);
            }
        }, 30000); // 30 seconds
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
