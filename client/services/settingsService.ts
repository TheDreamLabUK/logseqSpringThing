import { API_ENDPOINTS } from '../core/constants';
import { EventEmitter } from '../utils/eventEmitter';
import { Settings } from '../types/settings';

interface SettingsChangeEvent {
    category: string;
    setting: string;
    value: any;
}

type SettingsServiceEvents = {
    settingsChanged: SettingsChangeEvent;
};

export class SettingsService extends EventEmitter<SettingsServiceEvents> {
    private static instance: SettingsService;

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
        const currentSettings = await this.getSettings();
        
        // Handle nested settings structure
        const [mainCategory, subCategory] = category.split('.');
        if (subCategory) {
            if (!currentSettings[mainCategory as keyof Settings]) {
                throw new Error(`Invalid category: ${mainCategory}`);
            }
            (currentSettings[mainCategory as keyof Settings] as any)[subCategory][setting] = value;
        } else {
            if (!(category in currentSettings)) {
                throw new Error(`Invalid category: ${category}`);
            }
            (currentSettings[category] as any)[setting] = value;
        }

        // Send the full settings object
        const response = await fetch(API_ENDPOINTS.SETTINGS_ROOT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentSettings)
        });

        if (!response.ok) {
            throw new Error('Failed to update setting');
        }
        
        this.emit('settingsChanged', { category, setting, value });
    }

    async getSettings(): Promise<Settings> {
        const response = await fetch(API_ENDPOINTS.SETTINGS_ROOT);
        if (!response.ok) {
            throw new Error('Failed to fetch settings');
        }
        return response.json();
    }
}

export const settingsService = SettingsService.getInstance(); 