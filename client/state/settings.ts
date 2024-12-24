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
    private settings: Settings = defaultSettings;

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
            logger.error('Failed to initialize settings manager:', error);
            throw error;
        }
    }

    private checkInitialized(): void {
        if (!this.initialized) {
            logger.error('Settings manager not initialized');
            throw new Error('Settings manager must be initialized before use');
        }
    }

    public getCurrentSettings(): Settings {
        // Allow access to default settings even if not initialized
        return this.settings;
    }

    public async updateSetting(path: SettingsPath, value: SettingValue): Promise<void> {
        this.checkInitialized();
        
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }

        try {
            setSettingValue(this.settings, path, value);
            await this.store.set(path, value);
            logger.debug(`Updated setting ${path} to ${value}`);
        } catch (error) {
            logger.error(`Failed to update setting ${path}:`, error);
            throw error;
        }
    }

    public get(path: SettingsPath): SettingValue {
        // Allow access to default settings even if not initialized
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }
        
        try {
            return getSettingValue(this.settings, path);
        } catch (error) {
            logger.error(`Error getting setting at path ${path}:`, error);
            throw error;
        }
    }

    public getCategory(category: SettingsCategory): Settings[typeof category] {
        // Allow access to default settings even if not initialized
        if (!(category in this.settings)) {
            throw new Error(`Invalid settings category: ${category}`);
        }
        return this.settings[category];
    }

    public subscribe(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        this.checkInitialized();

        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }

        try {
            // Immediately notify with current value
            const currentValue = this.get(path);
            try {
                callback(currentValue);
            } catch (error) {
                logger.error(`Error in initial settings callback for ${path}:`, error);
            }

            return this.store.subscribe(path, (_, value) => {
                try {
                    callback(value as SettingValue);
                } catch (error) {
                    logger.error(`Error in settings subscriber for ${path}:`, error);
                }
            });
        } catch (error) {
            logger.error(`Error setting up subscription for ${path}:`, error);
            throw error;
        }
    }

    public onSettingChange(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        return this.subscribe(path, callback);
    }

    public async batchUpdate(updates: Array<{ path: SettingsPath; value: SettingValue }>): Promise<void> {
        this.checkInitialized();

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

            // Then sync with store
            await Promise.all(
                updates.map(({ path, value }) => this.store.set(path, value))
            );
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
