import { Settings, SettingCategory } from '../core/types';
import { createLogger } from '../core/logger';
import { SettingsStore } from './SettingsStore';
import { defaultSettings } from './defaultSettings';

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

    public async updateSetting<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        value: Settings[T][K]
    ): Promise<void> {
        this.settings[category][key] = value;
        this.set(category, key, value);
    }

    public get<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K
    ): Settings[T][K] {
        return this.store.get(`${category}.${key as string & K}`);
    }

    public set<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        value: Settings[T][K]
    ): void {
        this.store.set(`${category}.${key as string & K}`, value);
    }

    public getCategory<T extends keyof Settings>(category: T): Settings[T] {
        return this.store.getCategory(category);
    }

    public subscribe<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        callback: (value: Settings[T][K]) => void
    ): () => void {
        return this.store.subscribe(
            `${category}.${key as string & K}`,
            (_, value) => callback(value)
        );
    }

    public onSettingChange<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        callback: (value: Settings[T][K]) => void
    ): () => void {
        return this.subscribe(category, key, callback);
    }

    public dispose(): void {
        this.store.dispose();
        this.initialized = false;
    }
}

// Export singleton instance
export const settingsManager = new SettingsManager();
