import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { defaultSettings } from './defaultSettings';
import { buildApiUrl } from '../core/api';
import { API_ENDPOINTS } from '../core/constants';
import { Logger } from '../core/logger';
import { validateSettings, validateSettingValue, ValidationError } from '../types/settings/validation';

const logger = createLogger('SettingsStore');

export type SettingsChangeCallback = (path: string, value: unknown) => void;
export type ValidationErrorCallback = (errors: ValidationError[]) => void;

export class SettingsStore {
    private static instance: SettingsStore | null = null;
    private settings: Settings;
    private initialized: boolean = false;
    private initializationPromise: Promise<void> | null = null;
    private subscribers: Map<string, SettingsChangeCallback[]> = new Map();
    private validationSubscribers: ValidationErrorCallback[] = [];
    private logger: Logger;
    private retryCount: number = 0;
    private MAX_RETRIES: number = 3;
    private RETRY_DELAY: number = 1000;

    private constructor() {
        this.settings = {} as Settings;
        this.subscribers = new Map();
        this.logger = createLogger('SettingsStore');
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

                // Validate default settings
                const validationResult = validateSettings(this.settings);
                if (!validationResult.isValid) {
                    this.logger.error('Default settings validation failed:', validationResult.errors);
                    this.notifyValidationErrors(validationResult.errors);
                }

                // Try to fetch settings from server
                try {
                    logger.info('Fetching settings from:', buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT));
                    const response = await fetch(buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT));
                    logger.info('Server response status:', response.status);
                    
                    if (response.ok) {
                        const serverSettings = await response.json();
                        logger.info('Received server settings:', serverSettings);
                        
                        // Validate server settings before merging
                        const serverValidation = validateSettings(serverSettings);
                        if (!serverValidation.isValid) {
                            throw new Error(`Invalid server settings: ${JSON.stringify(serverValidation.errors)}`);
                        }
                        
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

                this.initialized = true;
                logger.info('SettingsStore initialized');
            } catch (error) {
                logger.error('Failed to initialize settings:', error);
                this.settings = { ...defaultSettings };
                this.initialized = true;
            }
        })();

        return this.initializationPromise;
    }

    public isInitialized(): boolean {
        return this.initialized;
    }

    public subscribeToValidationErrors(callback: ValidationErrorCallback): () => void {
        this.validationSubscribers.push(callback);
        return () => {
            const index = this.validationSubscribers.indexOf(callback);
            if (index > -1) {
                this.validationSubscribers.splice(index, 1);
            }
        };
    }

    public async subscribe(path: string, callback: SettingsChangeCallback): Promise<() => void> {
        if (!this.initialized) {
            await this.initialize();
        }

        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, []);
        }
        
        const subscribers = this.subscribers.get(path);
        if (subscribers) {
            subscribers.push(callback);
        }

        // Immediately call callback with current value
        const value = this.get(path);
        if (value !== undefined) {
            callback(path, value);
        }

        return () => {
            const pathSubscribers = this.subscribers.get(path);
            if (pathSubscribers) {
                const index = pathSubscribers.indexOf(callback);
                if (index > -1) {
                    pathSubscribers.splice(index, 1);
                }
                if (pathSubscribers.length === 0) {
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
        try {
            // Validate the specific setting change
            const validationErrors = validateSettingValue(path, value, this.settings);
            if (validationErrors.length > 0) {
                this.notifyValidationErrors(validationErrors);
                throw new Error(`Validation failed: ${JSON.stringify(validationErrors)}`);
            }
            
            // Create a copy of settings for rollback
            const previousSettings = JSON.parse(JSON.stringify(this.settings));
            
            // Update local state
            this.updateSettingValue(path, value);
            
            // Validate entire settings object after update
            const fullValidation = validateSettings(this.settings);
            if (!fullValidation.isValid) {
                // Rollback and notify of validation errors
                this.settings = previousSettings;
                this.notifyValidationErrors(fullValidation.errors);
                throw new Error(`Full validation failed: ${JSON.stringify(fullValidation.errors)}`);
            }
            
            // Sync with server
            try {
                await this.syncWithServer();
            } catch (error) {
                // Rollback on server sync failure
                this.settings = previousSettings;
                this.notifySubscribers(path, this.get(path));
                throw error;
            }
            
            // Notify subscribers of successful update
            this.notifySubscribers(path, value);
            
            this.logger.debug(`Setting updated successfully: ${path}`, value);
        } catch (error) {
            this.logger.error(`Failed to update setting: ${path}`, error);
            throw error;
        }
    }

    private async syncWithServer(): Promise<void> {
        try {
            const response = await fetch(buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.settings)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }
            
            // Update local settings with server response
            const serverSettings = await response.json();
            
            // Validate server response
            const validationResult = validateSettings(serverSettings);
            if (!validationResult.isValid) {
                throw new Error(`Invalid server response: ${JSON.stringify(validationResult.errors)}`);
            }
            
            this.settings = this.deepMerge(this.settings, serverSettings);
            this.logger.debug('Settings synced successfully with server');
        } catch (error) {
            this.logger.error('Failed to sync settings with server:', error);
            if (this.retryCount < this.MAX_RETRIES) {
                this.retryCount++;
                this.logger.info(`Retrying sync (attempt ${this.retryCount}/${this.MAX_RETRIES})...`);
                await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY));
                return this.syncWithServer();
            }
            throw error;
        }
    }

    private notifyValidationErrors(errors: ValidationError[]): void {
        this.validationSubscribers.forEach(callback => {
            try {
                callback(errors);
            } catch (error) {
                this.logger.error('Error in validation subscriber:', error);
            }
        });
    }

    private notifySubscribers(path: string, value: unknown): void {
        const subscribers = this.subscribers.get(path);
        if (subscribers) {
            subscribers.forEach(callback => {
                try {
                    callback(path, value);
                } catch (error) {
                    this.logger.error(`Error in settings subscriber for ${path}:`, error);
                }
            });
        }
    }

    private deepMerge(target: any, source: any): any {
        const result = { ...target };
        
        // Handle visualization section
        if (source.visualization) {
            result.visualization = result.visualization || {};
            for (const category in source.visualization) {
                if (result.visualization[category]) {
                    result.visualization[category] = {
                        ...result.visualization[category],
                        ...source.visualization[category]
                    };
                }
            }
        }
        
        // Handle system section
        if (source.system) {
            result.system = result.system || {};
            for (const category in source.system) {
                if (result.system[category]) {
                    result.system[category] = {
                        ...result.system[category],
                        ...source.system[category]
                    };
                }
            }
        }

        // Handle XR section
        if (source.xr) {
            result.xr = {
                ...result.xr,
                ...source.xr
            };
        }

        // Handle any other top-level properties
        for (const key in source) {
            if (!['visualization', 'system', 'xr'].includes(key)) {
                if (source[key] instanceof Object && key in target) {
                    result[key] = this.deepMerge(target[key], source[key]);
                } else {
                    result[key] = source[key];
                }
            }
        }
        
        return result;
    }

    private updateSettingValue(path: string, value: unknown): void {
        if (!path) {
            throw new Error('Setting path cannot be empty');
        }
        
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
    }

    public dispose(): void {
        this.subscribers.clear();
        this.validationSubscribers = [];
        this.settings = {} as Settings;
        SettingsStore.instance = null;
    }
}
