import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { buildApiUrl } from '../core/api';
import { defaultSettings } from './defaultSettings';
import { SettingsPath, SettingValue, SettingsCategory } from '../types/settings/utils';

const logger = createLogger('SettingsStore');

type SettingsChangeCallback = (path: SettingsPath, value: SettingValue) => void;

interface SettingsStoreOptions {
    autoSave?: boolean;
    syncInterval?: number;
}

export class SettingsStore {
    private static instance: SettingsStore;
    private settings: Settings;
    private subscribers: Map<SettingsPath, Set<SettingsChangeCallback>>;
    private pendingChanges: Set<SettingsPath>;
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
        const categories: SettingsCategory[] = ['visualization', 'xr', 'system'];
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
                        this.settings[category] = data.settings;
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
                const [category, ...rest] = path.split('.');
                const value = this.get(path);

                const response = await fetch(
                    buildApiUrl(`visualization/settings/${category}/${rest.join('/')}`),
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
            changes.forEach(path => this.pendingChanges.add(path));
        }
    }

    subscribe(path: SettingsPath, callback: SettingsChangeCallback): () => void {
        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path)!.add(callback);
        return () => this.unsubscribe(path, callback);
    }

    unsubscribe(path: SettingsPath, callback: SettingsChangeCallback): void {
        const callbacks = this.subscribers.get(path);
        if (callbacks) {
            callbacks.delete(callback);
            if (callbacks.size === 0) {
                this.subscribers.delete(path);
            }
        }
    }

    private notifySubscribers(path: SettingsPath, value: SettingValue): void {
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

    get(path: SettingsPath): SettingValue {
        const parts = path.split('.');
        return parts.reduce((obj: any, key) => obj && obj[key], this.settings);
    }

    set(path: SettingsPath, value: SettingValue): void {
        const parts = path.split('.');
        const lastKey = parts.pop()!;
        const target = parts.reduce((obj: any, key) => obj && obj[key], this.settings);
        
        if (target) {
            target[lastKey] = value;
            this.pendingChanges.add(path);
            this.notifySubscribers(path, value);
        }
    }

    getCategory(category: SettingsCategory): Settings[typeof category] {
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
