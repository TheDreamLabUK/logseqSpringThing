import { Settings } from '../core/types';
import { createLogger } from '../utils/logger';
import { defaultSettings } from './defaultSettings';
import { buildApiUrl } from '../core/api';

const logger = createLogger('SettingsManager');

interface Subscriber<T extends keyof Settings, K extends keyof Settings[T]> {
    callback: (value: Settings[T][K]) => void;
}

export class SettingsManager {
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
            const categories = Object.keys(this.settings) as Array<keyof Settings>;
            
            for (const category of categories) {
                let retries = 0;
                while (retries < maxRetries) {
                    try {
                        const response = await fetch(buildApiUrl(`visualization/settings/${category}`));
                        
                        if (response.ok) {
                            const data = await response.json();
                            if (data.success && data.settings) {
                                // Type-safe update of settings
                                const currentSettings = { ...this.settings[category] };
                                const newSettings = data.settings as Partial<Settings[typeof category]>;
                                
                                // Merge settings in a type-safe way
                                Object.keys(newSettings).forEach(key => {
                                    const settingKey = key as keyof Settings[typeof category];
                                    const value = newSettings[settingKey];
                                    if (value !== undefined) {
                                        (currentSettings as any)[settingKey] = value;
                                    }
                                });
                                
                                // Update the settings object
                                this.settings = {
                                    ...this.settings,
                                    [category]: currentSettings
                                };
                                
                                // Notify subscribers
                                Object.keys(newSettings).forEach(key => {
                                    const settingKey = key as keyof Settings[typeof category];
                                    const value = newSettings[settingKey];
                                    if (value !== undefined) {
                                        this.notifySubscribers(
                                            category,
                                            settingKey,
                                            value as Settings[typeof category][typeof settingKey]
                                        );
                                    }
                                });
                                
                                logger.info(`Loaded settings for category ${category}`);
                                break;
                            }
                        } else if (response.status === 404) {
                            logger.info(`Settings endpoint for ${category} not found, using defaults`);
                            break;
                        } else {
                            throw new Error(`Failed to fetch ${category} settings: ${response.statusText}`);
                        }
                    } catch (error) {
                        retries++;
                        if (retries === maxRetries) {
                            logger.error(`Failed to load ${category} settings after ${maxRetries} attempts:`, error);
                            break;
                        }
                        await new Promise(resolve => setTimeout(resolve, retryDelay));
                    }
                }
            }
            
            this.initialized = true;
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            throw error;
        }
    }

    public async updateSetting<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        value: Settings[T][K]
    ): Promise<void> {
        try {
            // Create a partial update for the category
            const update = {
                settings: {
                    [key]: value
                }
            };

            const response = await fetch(buildApiUrl(`visualization/settings/${category}`), {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(update),
            });

            if (!response.ok) {
                throw new Error(`Failed to update setting: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.success) {
                // Update local state safely
                this.settings = {
                    ...this.settings,
                    [category]: {
                        ...this.settings[category],
                        ...data.settings
                    }
                };
                
                // Notify subscribers of all updated settings
                Object.keys(data.settings).forEach(settingKey => {
                    const updatedKey = settingKey as K;
                    const updatedValue = data.settings[updatedKey];
                    if (updatedValue !== undefined) {
                        this.notifySubscribers(category, updatedKey, updatedValue as Settings[T][K]);
                    }
                });
            } else {
                throw new Error(data.error || 'Failed to update setting');
            }
        } catch (error) {
            logger.error(`Failed to update setting ${String(key)} in ${category}:`, error);
            throw error;
        }
    }

    public async updateCategorySettings<T extends keyof Settings>(
        category: T,
        settings: Partial<Settings[T]>
    ): Promise<void> {
        try {
            const update = {
                settings
            };

            const response = await fetch(buildApiUrl(`visualization/settings/${category}`), {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(update),
            });

            if (!response.ok) {
                throw new Error(`Failed to update category settings: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.success) {
                // Update local state with all returned settings
                this.settings = {
                    ...this.settings,
                    [category]: {
                        ...this.settings[category],
                        ...data.settings
                    }
                };
                // Notify subscribers of all updated settings
                Object.keys(data.settings).forEach(setting => {
                    this.notifySubscribers(category, setting as keyof Settings[T], data.settings[setting]);
                });
            } else {
                throw new Error(data.error || 'Failed to update category settings');
            }
        } catch (error) {
            logger.error(`Failed to update settings for category ${category}:`, error);
            throw error;
        }
    }

    private notifySubscribers<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        setting: K,
        value: Settings[T][K]
    ): void {
        const key = `${category}.${String(setting)}`;
        const subscribers = this.subscribers.get(key) || [];
        subscribers.forEach(sub => {
            try {
                sub.callback(value);
            } catch (error) {
                logger.error(`Error in subscriber callback for ${key}:`, error);
            }
        });
    }

    public subscribe<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        setting: K,
        callback: (value: Settings[T][K]) => void
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

    public getCurrentSettings(): Settings {
        return this.settings;
    }

    public getDefaultSettings(): Settings {
        return this.settings;
    }
}

// Re-export Settings interface
export type { Settings } from '../core/types';

// Initialize settings from settings.toml
export const settingsManager = new SettingsManager();
