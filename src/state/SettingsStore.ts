import { Settings } from '../types/settings';

export class SettingsStore {
    private static instance: SettingsStore;
    private settings: Settings | null = null;

    private constructor() {}

    static getInstance(): SettingsStore {
        if (!SettingsStore.instance) {
            SettingsStore.instance = new SettingsStore();
        }
        return SettingsStore.instance;
    }

    async initialize(): Promise<void> {
        // Implementation needed
    }

    get(key: string): Settings | null {
        return this.settings;
    }
} 