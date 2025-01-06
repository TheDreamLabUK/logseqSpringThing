import { Settings } from '../types/settings';
import { defaultSettings } from './defaultSettings';
import { createLogger } from '../core/logger';

export class SettingsStore {
    private static instance: SettingsStore;
    private settings: Settings;
    private logger = createLogger('SettingsStore');

    private constructor() {
        this.settings = { ...defaultSettings };
    }

    static getInstance(): SettingsStore {
        if (!SettingsStore.instance) {
            SettingsStore.instance = new SettingsStore();
        }
        return SettingsStore.instance;
    }

    async initialize(): Promise<void> {
        this.settings = { ...defaultSettings };
        this.logger.info('Using default settings');
        return Promise.resolve();
    }

    get(key?: string): Settings | any {
        if (!key) return this.settings;
        
        try {
            return key.split('.').reduce((obj: any, k) => obj[k], this.settings);
        } catch (error) {
            this.logger.error(`Error accessing setting at path ${key}:`, error);
            return undefined;
        }
    }
} 