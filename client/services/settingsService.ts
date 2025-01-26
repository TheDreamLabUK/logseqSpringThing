import { API_ENDPOINTS } from '../core/constants';
import { EventEmitter } from '../utils/eventEmitter';
import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';

const logger = createLogger('SettingsService');

interface SettingsChangeEvent {
    category: string;
    setting: string;
    value: any;
}

type SettingsServiceEvents = {
    settingsChanged: SettingsChangeEvent;
    error: Error;
};

export class SettingsService extends EventEmitter<SettingsServiceEvents> {
    private static instance: SettingsService;
    private currentSettings: Settings | null = null;
    private updateQueue: Promise<void> = Promise.resolve();

    private constructor() {
        super();
    }

    public static getInstance(): SettingsService {
        if (!SettingsService.instance) {
            SettingsService.instance = new SettingsService();
        }
        return SettingsService.instance;
    }

    async updateSetting(category: keyof Settings, setting: string, value: any): Promise<void> {
        // Queue updates to prevent race conditions
        this.updateQueue = this.updateQueue.then(() => this.performUpdate(category, setting, value));
        await this.updateQueue;
    }

    private async performUpdate(category: keyof Settings, setting: string, value: any): Promise<void> {
        try {
            logger.debug(`Updating setting: ${category}.${setting}`, value);

            // Ensure we have current settings
            if (!this.currentSettings) {
                this.currentSettings = await this.getSettings();
            }

            // Handle nested settings structure
            const [mainCategory, subCategory] = category.split('.');
            if (subCategory) {
                if (!this.currentSettings[mainCategory as keyof Settings]) {
                    throw new Error(`Invalid category: ${mainCategory}`);
                }
                (this.currentSettings[mainCategory as keyof Settings] as any)[subCategory][setting] = value;
            } else {
                if (!(category in this.currentSettings)) {
                    throw new Error(`Invalid category: ${category}`);
                }
                (this.currentSettings[category] as any)[setting] = value;
            }

            // Send the full settings object
            const response = await fetch(API_ENDPOINTS.SETTINGS_ROOT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.currentSettings)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to update setting: ${errorText}`);
            }

            // Update local settings with server response
            const updatedSettings = await response.json();
            this.currentSettings = updatedSettings;

            logger.debug(`Setting updated successfully: ${category}.${setting}`, value);
            this.emit('settingsChanged', { category, setting, value });
        } catch (error) {
            logger.error(`Failed to update setting: ${category}.${setting}`, error);
            // Reset current settings to force fresh fetch on next update
            this.currentSettings = null;
            // Re-throw error after emitting
            this.emit('error', error as Error);
            throw error;
        }
    }

    async getSettings(): Promise<Settings> {
        try {
            logger.debug('Fetching settings');
            const response = await fetch(API_ENDPOINTS.SETTINGS_ROOT);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch settings: ${errorText}`);
            }

            const settings = await response.json();
            this.currentSettings = settings;
            logger.debug('Settings fetched successfully');
            return settings;
        } catch (error) {
            logger.error('Failed to fetch settings:', error);
            this.currentSettings = null;
            this.emit('error', error as Error);
            throw error;
        }
    }

    // Helper method to validate setting path
    private validateSettingPath(category: string, setting: string): void {
        if (!this.currentSettings) {
            throw new Error('Settings not initialized');
        }

        const [mainCategory, subCategory] = category.split('.');
        if (subCategory) {
            if (!this.currentSettings[mainCategory as keyof Settings]) {
                throw new Error(`Invalid category: ${mainCategory}`);
            }
            const categoryObj = this.currentSettings[mainCategory as keyof Settings] as any;
            if (!categoryObj[subCategory]) {
                throw new Error(`Invalid subcategory: ${subCategory}`);
            }
            if (!(setting in categoryObj[subCategory])) {
                throw new Error(`Invalid setting: ${setting} in ${category}`);
            }
        } else {
            if (!(category in this.currentSettings)) {
                throw new Error(`Invalid category: ${category}`);
            }
            const categoryObj = this.currentSettings[category as keyof Settings] as any;
            if (!(setting in categoryObj)) {
                throw new Error(`Invalid setting: ${setting} in ${category}`);
            }
        }
    }
}

export const settingsService = SettingsService.getInstance();