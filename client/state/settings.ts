import { createLogger } from '../utils/logger';
import { Settings } from '../core/types';
import { convertObjectKeysToSnakeCase, convertObjectKeysToCamelCase } from '../core/utils';

const logger = createLogger('SettingsManager');

export class SettingsManager {
    private settings: Settings;
    private subscribers: Map<string, Map<string, Set<(value: any) => void>>> = new Map();

    constructor() {
        this.settings = this.getDefaultSettings();
        this.initializeSettings();
    }

    private getDefaultSettings(): Settings {
        // ... (keep existing default settings)
        return this.settings;
    }

    private async initializeSettings(): Promise<void> {
        try {
            await this.loadAllSettings();
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            // Keep default settings if loading fails
        }
    }

    public async updateSetting(category: keyof Settings, setting: string, value: any): Promise<void> {
        try {
            // Update local settings first
            const categorySettings = this.settings[category] as Record<string, any>;
            const oldValue = categorySettings[setting];
            categorySettings[setting] = value;

            // Convert category and setting to snake_case for API
            const snakeCaseCategory = convertObjectKeysToSnakeCase({ [category]: null })[0];
            const snakeCaseSetting = convertObjectKeysToSnakeCase({ [setting]: null })[0];

            // Prepare the request body with snake_case
            const requestBody = {
                value: convertObjectKeysToSnakeCase(value)
            };

            // Send update to server
            const response = await fetch(
                `/api/visualization/${snakeCaseCategory}/${snakeCaseSetting}`,
                {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                }
            );

            if (!response.ok) {
                // Revert local change on error
                categorySettings[setting] = oldValue;
                throw new Error(`Failed to update setting: ${response.statusText}`);
            }

            // Notify subscribers of successful update
            this.notifySubscribers(category, setting, value);
        } catch (error) {
            logger.error('Error updating setting:', error);
            throw error;
        }
    }

    public async getSetting(category: keyof Settings, setting: string): Promise<any> {
        try {
            // Convert camelCase to snake_case for API request
            const snakeCaseCategory = convertObjectKeysToSnakeCase({ [category]: null })[0];
            const snakeCaseSetting = convertObjectKeysToSnakeCase({ [setting]: null })[0];

            const response = await fetch(`/api/visualization/${snakeCaseCategory}/${snakeCaseSetting}`);
            if (!response.ok) {
                throw new Error(`Failed to get setting: ${response.statusText}`);
            }

            const data = await response.json();
            const value = convertObjectKeysToCamelCase(data.value);

            // Update local setting
            const categorySettings = this.settings[category] as Record<string, any>;
            categorySettings[setting] = value;

            return value;
        } catch (error) {
            logger.error('Error getting setting:', error);
            throw error;
        }
    }

    public async loadAllSettings(): Promise<void> {
        try {
            const response = await fetch('/api/visualization/settings');
            if (!response.ok) {
                throw new Error(`Failed to load settings: ${response.statusText}`);
            }

            const data = await response.json();
            const camelCaseSettings = convertObjectKeysToCamelCase(data);
            this.updateSettingsFromServer(camelCaseSettings);
        } catch (error) {
            logger.error('Error loading settings:', error);
            throw error;
        }
    }

    public async updateAllSettings(settings: Settings): Promise<void> {
        try {
            // Store old settings in case of error
            const oldSettings = this.getCurrentSettings();

            // Update local settings first
            this.settings = settings;

            const response = await fetch('/api/visualization/settings', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(convertObjectKeysToSnakeCase(settings))
            });

            if (!response.ok) {
                // Revert local changes on error
                this.settings = oldSettings;
                throw new Error(`Failed to update all settings: ${response.statusText}`);
            }

            const updatedSettings = convertObjectKeysToCamelCase(await response.json());
            this.updateSettingsFromServer(updatedSettings);
        } catch (error) {
            logger.error('Error updating all settings:', error);
            throw error;
        }
    }

    private updateSettingsFromServer(newSettings: Settings): void {
        // Store old settings for comparison
        const oldSettings = this.settings;
        this.settings = newSettings;

        // Compare and notify only changed settings
        Object.entries(newSettings).forEach(([category, categorySettings]) => {
            if (typeof categorySettings === 'object') {
                Object.entries(categorySettings).forEach(([setting, value]) => {
                    const oldValue = (oldSettings[category as keyof Settings] as any)?.[setting];
                    if (JSON.stringify(value) !== JSON.stringify(oldValue)) {
                        this.notifySubscribers(category, setting, value);
                    }
                });
            }
        });

        // Emit a global settings change event
        this.notifyGlobalSettingsChange(newSettings);
    }

    private notifyGlobalSettingsChange(settings: Settings): void {
        // Dispatch a custom event that components can listen to for global settings changes
        const event = new CustomEvent('settingsChanged', { detail: settings });
        window.dispatchEvent(event);
    }

    private notifyAllSubscribers(): void {
        Object.entries(this.settings).forEach(([category, categorySettings]) => {
            if (typeof categorySettings === 'object') {
                Object.entries(categorySettings).forEach(([setting, value]) => {
                    this.notifySubscribers(category, setting, value);
                });
            }
        });
    }

    private notifySubscribers<T>(category: string, setting: string, value: T): void {
        const categoryMap = this.subscribers.get(category);
        if (!categoryMap) return;

        const settingSet = categoryMap.get(setting);
        if (!settingSet) return;

        settingSet.forEach(listener => {
            try {
                listener(value);
            } catch (error) {
                logger.error(`Error in settings listener for ${category}.${setting}:`, error);
            }
        });
    }

    public subscribe<T>(category: string, setting: string, listener: (value: T) => void): () => void {
        if (!this.subscribers.has(category)) {
            this.subscribers.set(category, new Map());
        }
        const categoryMap = this.subscribers.get(category)!;
        
        if (!categoryMap.has(setting)) {
            categoryMap.set(setting, new Set());
        }
        const settingSet = categoryMap.get(setting)!;
        
        settingSet.add(listener);
        
        // Send current value immediately
        const currentValue = (this.settings[category as keyof Settings] as any)[setting];
        if (currentValue !== undefined) {
            try {
                listener(currentValue);
            } catch (error) {
                logger.error(`Error in initial settings listener for ${category}.${setting}:`, error);
            }
        }
        
        return () => {
            settingSet.delete(listener);
            if (settingSet.size === 0) {
                categoryMap.delete(setting);
            }
            if (categoryMap.size === 0) {
                this.subscribers.delete(category);
            }
        };
    }

    public getCurrentSettings(): Settings {
        return JSON.parse(JSON.stringify(this.settings));
    }

    public dispose(): void {
        this.subscribers.clear();
    }
}

// Create singleton instance
export const settingsManager = new SettingsManager();

// Re-export Settings interface
export type { Settings };
