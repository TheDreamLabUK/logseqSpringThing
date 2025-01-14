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
        this.store = SettingsStore.getInstance();
    }

    private useDefaultSettings(): void {
        // Reset to default settings
        this.settings = { ...defaultSettings };
        this.initialized = true;
    }

    public async initialize(): Promise<void> {
        if (this.initialized) return;

        try {
            await this.store.initialize();
            this.settings = this.store.get('') as Settings;
            this.initialized = true;
            logger.info('Settings initialized from server');
        } catch (error) {
            logger.error('Failed to initialize settings from server:', error);
            this.useDefaultSettings();
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

    public subscribe(path: string, callback: (value: unknown) => void): () => void {
        const store = SettingsStore.getInstance();
        let unsubscriber: (() => void) | undefined;
        
        store.subscribe(path, (_, value) => {
            callback(value);
        }).then(unsub => {
            unsubscriber = unsub;
        });

        return () => {
            if (unsubscriber) {
                unsubscriber();
            }
        };
    }

    public onSettingChange(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        const store = SettingsStore.getInstance();
        let unsubscriber: (() => void) | undefined;
        
        store.subscribe(path, (_, value) => {
            callback(value as SettingValue);
        }).then(unsub => {
            unsubscriber = unsub;
        });

        return () => {
            if (unsubscriber) {
                unsubscriber();
            }
        };
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
