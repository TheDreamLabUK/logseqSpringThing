import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { SettingsStore } from './SettingsStore';
import { defaultSettings } from './defaultSettings';
import {
    SettingsCategory,
    SettingsPath,
    SettingValue,
    getSettingValue,
    setSettingValue,
    isValidSettingPath
} from '../types/settings/utils';

const logger = createLogger('SettingsManager');

export class SettingsManager {
    private store: SettingsStore;
    private initialized: boolean = false;
    private settings: Settings = { ...defaultSettings };

    constructor() {
        this.store = SettingsStore.getInstance({ autoSave: true });
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            await this.store.initialize();
            this.initialized = true;
            logger.info('Settings manager initialized');
        } catch (error) {
            // Log error but continue with default settings
            logger.error('Failed to initialize settings manager:', error);
            logger.info('Continuing with default settings');
            this.initialized = true;
        }
    }

    public getCurrentSettings(): Settings {
        // Always return settings, which will be defaults if initialization failed
        return this.settings;
    }

    public async updateSetting(path: SettingsPath, value: SettingValue): Promise<void> {
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }

        try {
            setSettingValue(this.settings, path, value);
            if (this.initialized) {
                await this.store.set(path, value);
            } else {
                logger.warn(`Setting ${path} updated in memory only - store not initialized`);
            }
            logger.debug(`Updated setting ${path} to ${value}`);
        } catch (error) {
            logger.error(`Failed to update setting ${path}:`, error);
            throw error;
        }
    }

    public get(path: SettingsPath): SettingValue {
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }
        
        try {
            return getSettingValue(this.settings, path);
        } catch (error) {
            logger.error(`Error getting setting at path ${path}:`, error);
            // Return default value for this path if available
            return getSettingValue(defaultSettings, path);
        }
    }

    public getCategory(category: SettingsCategory): Settings[typeof category] {
        if (!(category in this.settings)) {
            logger.warn(`Category ${category} not found, using defaults`);
            return defaultSettings[category];
        }
        return this.settings[category];
    }

    public subscribe(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }

        try {
            // Get current value for immediate notification
            const currentValue = this.get(path);
            try {
                callback(currentValue);
            } catch (error) {
                logger.error(`Error in initial settings callback for ${path}:`, error);
            }

            // Set up subscription if store is initialized
            if (this.initialized) {
                return this.store.subscribe(path, (_, value) => {
                    try {
                        callback(value as SettingValue);
                    } catch (error) {
                        logger.error(`Error in settings subscriber for ${path}:`, error);
                    }
                });
            } else {
                logger.warn(`Subscription for ${path} not set up - store not initialized`);
                return () => {}; // Return no-op cleanup function
            }
        } catch (error) {
            logger.error(`Error setting up subscription for ${path}:`, error);
            throw error;
        }
    }

    public onSettingChange(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        return this.subscribe(path, callback);
    }

    public async batchUpdate(updates: Array<{ path: SettingsPath; value: SettingValue }>): Promise<void> {
        try {
            // Validate all paths first
            for (const { path } of updates) {
                if (!isValidSettingPath(path)) {
                    throw new Error(`Invalid settings path: ${path}`);
                }
            }

            // Apply updates to local settings first
            for (const { path, value } of updates) {
                setSettingValue(this.settings, path, value);
            }

            // Then sync with store if initialized
            if (this.initialized) {
                await Promise.all(
                    updates.map(({ path, value }) => this.store.set(path, value))
                );
            } else {
                logger.warn('Settings updated in memory only - store not initialized');
            }
        } catch (error) {
            logger.error('Failed to apply batch updates:', error);
            throw error;
        }
    }

    public dispose(): void {
        this.store.dispose();
        this.initialized = false;
    }
}

// Export singleton instance
export const settingsManager = new SettingsManager();
