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
    isValidSettingPath,
    getSettingCategory
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

    public getCurrentSettings(): Settings {
        return this.settings;
    }

    public async updateSetting(path: SettingsPath, value: SettingValue): Promise<void> {
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
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }
        return getSettingValue(this.settings, path);
    }

    public getCategory(category: SettingsCategory): Settings[typeof category] {
        return this.settings[category];
    }

    public subscribe(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        if (!isValidSettingPath(path)) {
            throw new Error(`Invalid settings path: ${path}`);
        }

        return this.store.subscribe(path, (_, value) => {
            try {
                callback(value);
            } catch (error) {
                logger.error(`Error in settings subscriber for ${path}:`, error);
            }
        });
    }

    public onSettingChange(path: SettingsPath, callback: (value: SettingValue) => void): () => void {
        return this.subscribe(path, callback);
    }

    public async batchUpdate(updates: Array<{ path: SettingsPath; value: SettingValue }>): Promise<void> {
        for (const { path, value } of updates) {
            if (!isValidSettingPath(path)) {
                throw new Error(`Invalid settings path: ${path}`);
            }
            setSettingValue(this.settings, path, value);
        }

        try {
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
