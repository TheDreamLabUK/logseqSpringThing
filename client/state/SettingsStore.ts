import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { defaultSettings } from './defaultSettings';
import { buildApiUrl } from '../core/api';
import { API_ENDPOINTS } from '../core/constants';

const logger = createLogger('SettingsStore');

export type SettingsChangeCallback = (path: string, value: unknown) => void;

export class SettingsStore {
    private static instance: SettingsStore | null = null;
    private settings: Settings;
    private initialized: boolean = false;
    private initializationPromise: Promise<void> | null = null;
    private subscribers: Map<string, Set<SettingsChangeCallback>> = new Map();

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

                // Try to fetch settings from server
                try {
                    logger.info('Fetching settings from:', buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT));
                    const response = await fetch(buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT));
                    logger.info('Server response status:', response.status);
                    
                    if (response.ok) {
                        const serverSettings = await response.json();
                        logger.info('Received server settings:', serverSettings);
                        
                        // Log current defaults before merge
                        logger.info('Current default settings:', this.settings);
                        
                        // Deep merge server settings with defaults
                        this.settings = this.deepMerge(this.settings, serverSettings);
                        logger.info('Merged settings:', this.settings);
                    } else {
                        const errorText = await response.text();
                        throw new Error(`Failed to fetch server settings: ${response.statusText}. Details: ${errorText}`);
                    }
                } catch (error) {
                    logger.warn('Error loading server settings:', error);
                    logger.info('Using default settings:', this.settings);
                }

                // Ensure critical settings are set
                if (this.settings.visualization?.physics) {
                    this.settings.visualization.physics.enabled = true;
                }
                
                // Validate settings structure
                if (!this.settings.visualization || !this.settings.system) {
                    logger.error('Invalid settings structure, resetting to defaults');
                    this.settings = { ...defaultSettings };
                }

                this.initialized = true;
                logger.info('SettingsStore initialized');
            } catch (error) {
                logger.error('Failed to initialize settings:', error);
                // Use defaults on error but ensure physics is enabled
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

    public async set(path: string, value: unknown): Promise<void> {
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
            
            // Immediately sync with server
            try {
                const response = await fetch(buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify([{
                        path,
                        value
                    }]),
                });

                if (!response.ok) {
                    throw new Error(`Failed to sync setting: ${response.statusText}`);
                }
                
                logger.debug(`Setting ${path} synced with server`);
            } catch (error) {
                logger.error(`Failed to sync setting ${path}:`, error);
                // Continue with local update even if server sync fails
            }

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
        
        // Handle visualization section specially since server sends it flat
        if (source.visualization) {
            for (const category in source.visualization) {
                if (result.visualization && result.visualization[category]) {
                    result.visualization[category] = {
                        ...result.visualization[category],
                        ...source.visualization[category]
                    };
                }
            }
        }
        
        // Handle system section
        if (source.system) {
            for (const category in source.system) {
                if (result.system && result.system[category]) {
                    result.system[category] = {
                        ...result.system[category],
                        ...source.system[category]
                    };
                }
            }
        }

        // Handle any other top-level properties
        for (const key in source) {
            if (key !== 'visualization' && key !== 'system') {
                if (source[key] instanceof Object && key in target) {
                    result[key] = this.deepMerge(target[key], source[key]);
                } else {
                    result[key] = source[key];
                }
            }
        }
        
        return result;
    }

    public dispose(): void {
        this.subscribers.clear();
        this.initialized = false;
    }
}
