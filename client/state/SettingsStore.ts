import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { defaultSettings } from './defaultSettings';
import { buildApiUrl } from '../core/api';
import { API_ENDPOINTS } from '../core/constants';
import { Logger } from '../core/logger';
import { validateSettings, validateSettingValue, ValidationError } from '../types/settings/validation';
import { convertObjectKeysToSnakeCase, convertObjectKeysToCamelCase } from '../core/utils';

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
    private settingsOrigin: 'server' | 'default' = 'default';

    private constructor() {
        this.settings = { ...defaultSettings };
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
                // Try to fetch settings from server first
                try {
                    logger.info('Fetching settings from:', buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT));
                    const response = await fetch(buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT));
                    logger.info('Server response status:', response.status);
                    
                    if (response.ok) {
                        const serverSettings = await response.json();
                        logger.info('Received server settings:', serverSettings);
                        
                        // Convert snake_case to camelCase
                        const camelCaseSettings = convertObjectKeysToCamelCase(serverSettings);
                        
                        // Validate server settings
                        const serverValidation = validateSettings(camelCaseSettings);
                        if (!serverValidation.isValid) {
                            throw new Error(`Invalid server settings: ${JSON.stringify(serverValidation.errors)}`);
                        }
                        
                        // Use server settings as base, filling in any missing fields with defaults
                        this.settings = this.deepMerge({ ...defaultSettings }, camelCaseSettings);
                        this.settingsOrigin = 'server';
                        logger.info('Using server settings with defaults as fallback');
                    } else {
                        const errorText = await response.text();
                        throw new Error(`Failed to fetch server settings: ${response.statusText}. Details: ${errorText}`);
                    }
                } catch (error) {
                    // If server settings fail, fall back to defaults
                    logger.warn('Error loading server settings, falling back to defaults:', error);
                    this.settings = { ...defaultSettings };
                    this.settingsOrigin = 'default';
                    
                    // Validate default settings
                    const validationResult = validateSettings(this.settings);
                    if (!validationResult.isValid) {
                        this.logger.error('Default settings validation failed:', validationResult.errors);
                        this.notifyValidationErrors(validationResult.errors);
                    }
                    
                    logger.info('Using default settings:', this.settings);
                }

                this.initialized = true;
                logger.info('SettingsStore initialized with origin:', this.settingsOrigin);
            } catch (error) {
                logger.error('Critical initialization failure:', error);
                // Last resort: use defaults without validation
                this.settings = { ...defaultSettings };
                this.settingsOrigin = 'default';
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
            const previousOrigin = this.settingsOrigin;
            
            // Update local state
            this.updateSettingValue(path, value);
            // Mark as modified from default since this is a user action
            this.settingsOrigin = 'default';
            
            // Validate entire settings object after update
            const fullValidation = validateSettings(this.settings);
            if (!fullValidation.isValid) {
                // Rollback and notify of validation errors
                this.settings = previousSettings;
                this.settingsOrigin = previousOrigin;
                this.notifyValidationErrors(fullValidation.errors);
                throw new Error(`Full validation failed: ${JSON.stringify(fullValidation.errors)}`);
            }
            
            // Sync with server (not an initial sync)
            try {
                await this.syncWithServer(false);
            } catch (error) {
                // Rollback on server sync failure
                this.settings = previousSettings;
                this.settingsOrigin = previousOrigin;
                this.notifySubscribers(path, this.get(path));
                throw error;
            }
            
            // Notify subscribers of successful update
            this.notifySubscribers(path, value);
            
            this.logger.debug(`Setting updated successfully: ${path}`, {
                value,
                origin: this.settingsOrigin
            });
        } catch (error) {
            this.logger.error(`Failed to update setting: ${path}`, error);
            throw error;
        }
    }

    public isFromServer(): boolean {
        return this.settingsOrigin === 'server';
    }

    private prepareSettingsForSync(settings: Settings): any {
        // Create a copy of settings
        const preparedSettings = JSON.parse(JSON.stringify(settings));

        // Ensure required sections exist
        if (!preparedSettings.system) preparedSettings.system = {};
        if (!preparedSettings.system.debug) preparedSettings.system.debug = {};
        if (!preparedSettings.xr) preparedSettings.xr = {};

        // Always include all required debug fields
        preparedSettings.system.debug = {
            enabled: preparedSettings.system.debug.enabled ?? false,
            enableDataDebug: preparedSettings.system.debug.enableDataDebug ?? false,
            enableWebsocketDebug: preparedSettings.system.debug.enableWebsocketDebug ?? false,
            logBinaryHeaders: preparedSettings.system.debug.logBinaryHeaders ?? false,
            logFullJson: preparedSettings.system.debug.logFullJson ?? false,
            logLevel: preparedSettings.system.debug.logLevel ?? 'info',
            logFormat: preparedSettings.system.debug.logFormat ?? 'json'
        };

        // Always include required XR fields
        const defaultXR = defaultSettings.xr;
        preparedSettings.xr = {
            ...preparedSettings.xr,
            gestureSmoothing: preparedSettings.xr.gestureSmoothing ?? defaultXR.gestureSmoothing,
            mode: preparedSettings.xr.mode ?? defaultXR.mode,
            roomScale: preparedSettings.xr.roomScale ?? defaultXR.roomScale,
            quality: preparedSettings.xr.quality ?? defaultXR.quality,
            spaceType: preparedSettings.xr.spaceType ?? defaultXR.spaceType,
            enableHandTracking: preparedSettings.xr.enableHandTracking ?? defaultXR.enableHandTracking,
            handMeshEnabled: preparedSettings.xr.handMeshEnabled ?? defaultXR.handMeshEnabled,
            handMeshColor: preparedSettings.xr.handMeshColor ?? defaultXR.handMeshColor,
            handMeshOpacity: preparedSettings.xr.handMeshOpacity ?? defaultXR.handMeshOpacity,
            handPointSize: preparedSettings.xr.handPointSize ?? defaultXR.handPointSize,
            handRayEnabled: preparedSettings.xr.handRayEnabled ?? defaultXR.handRayEnabled,
            handRayColor: preparedSettings.xr.handRayColor ?? defaultXR.handRayColor,
            handRayWidth: preparedSettings.xr.handRayWidth ?? defaultXR.handRayWidth,
            movementAxes: preparedSettings.xr.movementAxes ?? defaultXR.movementAxes
        };

        // Convert to snake_case for server
        return convertObjectKeysToSnakeCase(preparedSettings);
    }

    private async syncWithServer(isInitialSync: boolean = false): Promise<void> {
        // Don't sync to server during initialization if we got settings from server
        if (isInitialSync && this.settingsOrigin === 'server') {
            this.logger.debug('Skipping initial sync as settings came from server');
            return;
        }

        try {
            // Prepare settings for server sync
            const serverSettings = this.prepareSettingsForSync(this.settings);
            
            this.logger.debug('Sending settings to server:', {
                origin: this.settingsOrigin,
                isInitialSync,
                debug: serverSettings.system?.debug
            });
            
            const response = await fetch(buildApiUrl(API_ENDPOINTS.SETTINGS_ROOT), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(serverSettings)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                this.logger.error('Server sync failed:', {
                    status: response.status,
                    error: errorText,
                    sentSettings: serverSettings.system?.debug
                });
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }
            
            // Convert server response back to camelCase
            const responseData = await response.json();
            const camelCaseSettings = convertObjectKeysToCamelCase(responseData);
            
            this.logger.debug('Received settings from server:', {
                debug: camelCaseSettings.system?.debug
            });
            
            // Validate server response
            const validationResult = validateSettings(camelCaseSettings);
            if (!validationResult.isValid) {
                this.logger.error('Settings validation failed:', {
                    errors: validationResult.errors,
                    receivedSettings: camelCaseSettings.system?.debug
                });
                throw new Error(`Invalid server response: ${JSON.stringify(validationResult.errors)}`);
            }
            
            this.settings = this.deepMerge(this.settings, camelCaseSettings);
            this.logger.debug('Settings synced successfully:', {
                finalDebug: this.settings.system?.debug
            });
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
        
        // Handle arrays
        if (Array.isArray(source)) {
            return [...source];
        }
        
        if (source && typeof source === 'object') {
            Object.keys(source).forEach(key => {
                if (source[key] instanceof Object && !Array.isArray(source[key])) {
                    result[key] = this.deepMerge(result[key] || {}, source[key]);
                } else {
                    result[key] = source[key];
                }
            });
        }
        
        return result;
    }

    private updateSettingValue(path: string, value: unknown): void {
        if (!path) {
            throw new Error('Setting path cannot be empty');
        }
        
        const parts = path.split('.');
        const section = parts[0];
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

        // Update the specific value
        target[lastKey] = value;

        // If this is an XR setting, ensure all required fields are present
        if (section === 'xr') {
            const currentXR = this.settings.xr;
            const defaultXR = defaultSettings.xr;
            
            // Ensure all required XR fields are present with defaults
            this.settings.xr = {
                ...currentXR,
                mode: currentXR.mode ?? defaultXR.mode,
                roomScale: currentXR.roomScale ?? defaultXR.roomScale,
                spaceType: currentXR.spaceType ?? defaultXR.spaceType,
                quality: currentXR.quality ?? defaultXR.quality,
                gestureSmoothing: currentXR.gestureSmoothing ?? defaultXR.gestureSmoothing,
                enableHandTracking: currentXR.enableHandTracking ?? defaultXR.enableHandTracking,
                handMeshEnabled: currentXR.handMeshEnabled ?? defaultXR.handMeshEnabled,
                handMeshColor: currentXR.handMeshColor ?? defaultXR.handMeshColor,
                handMeshOpacity: currentXR.handMeshOpacity ?? defaultXR.handMeshOpacity,
                handPointSize: currentXR.handPointSize ?? defaultXR.handPointSize,
                handRayEnabled: currentXR.handRayEnabled ?? defaultXR.handRayEnabled,
                handRayColor: currentXR.handRayColor ?? defaultXR.handRayColor,
                handRayWidth: currentXR.handRayWidth ?? defaultXR.handRayWidth,
                movementAxes: currentXR.movementAxes ?? defaultXR.movementAxes
            };
        }
    }

    public dispose(): void {
        this.subscribers.clear();
        this.validationSubscribers = [];
        this.settings = { ...defaultSettings };
        SettingsStore.instance = null;
    }
}
