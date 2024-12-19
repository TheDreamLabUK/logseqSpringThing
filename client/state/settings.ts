import { Settings, SettingsManager as ISettingsManager, SettingCategory, SettingKey, SettingValueType } from '../types/settings';
import { defaultSettings } from './defaultSettings';
import { createLogger } from '../utils/logger';

const logger = createLogger('SettingsManager');

type Subscriber<T extends SettingCategory, K extends SettingKey<T>> = {
    callback: (value: SettingValueType<T, K>) => void;
};

class SettingsManager implements ISettingsManager {
    private settings: Settings;
    private subscribers: Map<string, Array<Subscriber<any, any>>> = new Map();
    private initialized: boolean = false;

    constructor() {
        this.settings = { ...defaultSettings };
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        const maxRetries = 3;
        const retryDelay = 1000; // 1 second

        try {
            const categories = Object.keys(this.settings) as SettingCategory[];
            
            for (const category of categories) {
                let retries = 0;
                while (retries < maxRetries) {
                    try {
                        const response = await fetch(`/api/visualization/settings/${category}`);
                        
                        if (response.ok) {
                            const data = await response.json();
                            if (this.settings[category]) {
                                this.settings[category] = { ...this.settings[category], ...data };
                                logger.info(`Loaded settings for category ${category}`);
                                break; // Success, exit retry loop
                            }
                        } else if (response.status === 404) {
                            logger.info(`Settings endpoint for ${category} not found, using defaults`);
                            break; // 404 is expected for some categories, exit retry loop
                        } else {
                            throw new Error(`Failed to fetch ${category} settings: ${response.statusText}`);
                        }
                    } catch (error) {
                        retries++;
                        if (retries === maxRetries) {
                            logger.error(`Failed to load ${category} settings after ${maxRetries} attempts:`, error);
                            logger.info(`Using default values for ${category} settings`);
                        } else {
                            logger.warn(`Retry ${retries}/${maxRetries} for ${category} settings`);
                            await new Promise(resolve => setTimeout(resolve, retryDelay));
                        }
                    }
                }
            }
            
            this.initialized = true;
            logger.info('Settings initialization complete');
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            throw error;
        }
    }

    public getCurrentSettings(): Settings {
        return this.settings;
    }

    public getDefaultSettings(): Settings {
        return defaultSettings;
    }

    public async updateSetting<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>
    ): Promise<void> {
        try {
            if (!(category in this.settings)) {
                throw new Error(`Invalid category: ${category}`);
            }

            const categorySettings = this.settings[category];
            if (!(String(setting) in categorySettings)) {
                throw new Error(`Invalid setting: ${String(setting)} in category ${category}`);
            }

            // Update the setting
            (this.settings[category] as any)[setting] = value;

            // Notify subscribers
            const key = `${category}.${String(setting)}`;
            const subscribers = this.subscribers.get(key) || [];
            subscribers.forEach(sub => {
                try {
                    sub.callback(value);
                } catch (error) {
                    logger.error(`Error in subscriber callback for ${key}:`, error);
                }
            });

            // Save settings to backend
            await this.saveSettings(category, setting, value);

        } catch (error) {
            logger.error(`Error updating setting ${category}.${String(setting)}:`, error);
            throw error;
        }
    }

    private async saveSettings<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>
    ): Promise<void> {
        try {
            const response = await fetch(`/api/visualization/settings/${category}/${String(setting)}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ value }),
            });

            if (!response.ok) {
                throw new Error(`Failed to save setting: ${response.statusText}`);
            }
        } catch (error) {
            logger.error(`Error saving setting ${category}.${String(setting)}:`, error);
            throw error;
        }
    }

    public subscribe<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        callback: (value: SettingValueType<T, K>) => void
    ): () => void {
        const key = `${category}.${String(setting)}`;
        if (!this.subscribers.has(key)) {
            this.subscribers.set(key, []);
        }

        const subscriber = { callback };
        this.subscribers.get(key)!.push(subscriber);

        return () => {
            const subscribers = this.subscribers.get(key);
            if (subscribers) {
                const index = subscribers.indexOf(subscriber);
                if (index !== -1) {
                    subscribers.splice(index, 1);
                }
            }
        };
    }

    public dispose(): void {
        this.subscribers.clear();
    }
}

export const settingsManager = new SettingsManager();

// Re-export Settings interface
export type { Settings } from '../types/settings';
