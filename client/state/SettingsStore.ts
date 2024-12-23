import { Settings } from '../core/types';
import { createLogger } from '../core/logger';
import { buildApiUrl } from '../core/api';
import { defaultSettings } from './defaultSettings';

const logger = createLogger('SettingsStore');

export type SettingPath = string;
export type SettingValue = any;
export type SettingsChangeCallback = (path: SettingPath, value: SettingValue) => void;

interface SettingsStoreOptions {
    autoSave?: boolean;
    syncInterval?: number;
}

export class SettingsStore {
    private static instance: SettingsStore;
    private settings: Settings;
    private subscribers: Map<SettingPath, Set<SettingsChangeCallback>>;
    private pendingChanges: Set<SettingPath>;
    private syncTimer: NodeJS.Timeout | null;
    private initialized: boolean;
    private options: Required<SettingsStoreOptions>;

    private constructor(options: SettingsStoreOptions = {}) {
        this.settings = { ...defaultSettings };
        this.subscribers = new Map();
        this.pendingChanges = new Set();
        this.syncTimer = null;
        this.initialized = false;
        this.options = {
            autoSave: options.autoSave ?? true,
            syncInterval: options.syncInterval ?? 5000
        };
    }

    static getInstance(options?: SettingsStoreOptions): SettingsStore {
        if (!SettingsStore.instance) {
            SettingsStore.instance = new SettingsStore(options);
        }
        return SettingsStore.instance;
    }

    async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            await this.loadAllSettings();
            if (this.options.autoSave) {
                this.startSync();
            }
            this.initialized = true;
        } catch (error) {
            logger.error('Failed to initialize settings store:', error);
            throw error;
        }
    }

    private async loadAllSettings(): Promise<void> {
        const categories = Object.keys(this.settings) as Array<keyof Settings>;
        const maxRetries = 3;
        const retryDelay = 1000;

        for (const category of categories) {
            let retries = 0;
            while (retries < maxRetries) {
                try {
                    const response = await fetch(buildApiUrl(`visualization/settings/${category}`));
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    if (data.success && data.settings) {
                        this.updateCategorySettings(category, data.settings);
                    }
                    break;
                } catch (error) {
                    retries++;
                    if (retries === maxRetries) {
                        logger.error(`Failed to load settings for ${category}:`, error);
                        throw error;
                    }
                    await new Promise(resolve => setTimeout(resolve, retryDelay));
                }
            }
        }
    }

    private updateCategorySettings<T extends keyof Settings>(
        category: T,
        newSettings: Partial<Settings[T]>
    ): void {
        const currentSettings = { ...this.settings[category] };
        let hasChanges = false;

        Object.entries(newSettings).forEach(([key, value]) => {
            const settingKey = key as keyof Settings[T];
            if (value !== undefined && value !== currentSettings[settingKey]) {
                (currentSettings as any)[settingKey] = value;
                hasChanges = true;
                this.notifySubscribers(`${category}.${key}`, value);
            }
        });

        if (hasChanges) {
            this.settings[category] = currentSettings;
        }
    }

    private startSync(): void {
        if (this.syncTimer) {
            return;
        }

        this.syncTimer = setInterval(() => {
            this.syncPendingChanges();
        }, this.options.syncInterval);
    }

    private async syncPendingChanges(): Promise<void> {
        if (this.pendingChanges.size === 0) {
            return;
        }

        const changes = Array.from(this.pendingChanges);
        this.pendingChanges.clear();

        try {
            const updatePromises = changes.map(async path => {
                const [category, key] = path.split('.');
                const value = this.get(path);

                const response = await fetch(
                    buildApiUrl(`visualization/settings/${category}/${key}`),
                    {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ value })
                    }
                );

                if (!response.ok) {
                    throw new Error(`Failed to sync setting ${path}`);
                }
            });

            await Promise.all(updatePromises);
        } catch (error) {
            logger.error('Error syncing settings:', error);
            // Re-add failed changes to pending changes
            changes.forEach(path => this.pendingChanges.add(path));
        }
    }

    subscribe(path: SettingPath, callback: SettingsChangeCallback): () => void {
        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path)!.add(callback);
        return () => this.unsubscribe(path, callback);
    }

    unsubscribe(path: SettingPath, callback: SettingsChangeCallback): void {
        const callbacks = this.subscribers.get(path);
        if (callbacks) {
            callbacks.delete(callback);
            if (callbacks.size === 0) {
                this.subscribers.delete(path);
            }
        }
    }

    private notifySubscribers(path: SettingPath, value: SettingValue): void {
        const callbacks = this.subscribers.get(path);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(path, value);
                } catch (error) {
                    logger.error(`Error in settings subscriber for ${path}:`, error);
                }
            });
        }
    }

    get<T extends keyof Settings, K extends keyof Settings[T]>(
        path: `${T}.${string & K}`
    ): Settings[T][K] {
        const [category, key] = path.split('.') as [T, K];
        return this.settings[category][key];
    }

    set<T extends keyof Settings, K extends keyof Settings[T]>(
        path: `${T}.${string & K}`,
        value: Settings[T][K]
    ): void {
        const [category, key] = path.split('.') as [T, K];
        if (this.settings[category][key] !== value) {
            this.settings[category][key] = value;
            this.pendingChanges.add(path);
            this.notifySubscribers(path, value);
        }
    }

    getCategory<T extends keyof Settings>(category: T): Settings[T] {
        return { ...this.settings[category] };
    }

    dispose(): void {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
            this.syncTimer = null;
        }
        this.subscribers.clear();
        this.pendingChanges.clear();
        this.initialized = false;
    }
}
