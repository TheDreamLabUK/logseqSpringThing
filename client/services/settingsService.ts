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

    async updateSetting(category: string, setting: string, value: any): Promise<void> {
        const response = await fetch(
            API_ENDPOINTS.SETTINGS_ITEM(category, setting), 
            {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ value })
            }
        );
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