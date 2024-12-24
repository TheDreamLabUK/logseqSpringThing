import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';

const logger = createLogger('SettingsStore');

export interface SettingsStoreOptions {
    autoSave?: boolean;
    syncInterval?: number;
}

const defaultSettingsStoreOptions: Required<SettingsStoreOptions> = {
    autoSave: true,
    syncInterval: 5000
};

export type SettingsChangeCallback = (path: string, value: unknown) => void;

export class SettingsStore {
    private static instance: SettingsStore | null = null;
    private settings: Settings;
    private initialized: boolean = false;
    private pendingChanges: Set<string> = new Set();
    private subscribers: Map<string, Set<SettingsChangeCallback>> = new Map();
    private syncTimer: number | null = null;

    private constructor(
        private readonly options: SettingsStoreOptions = defaultSettingsStoreOptions
    ) {
        this.settings = {} as Settings;
    }

    public static getInstance(options?: SettingsStoreOptions): SettingsStore {
        if (!SettingsStore.instance) {
            SettingsStore.instance = new SettingsStore(options);
        }
        return SettingsStore.instance;
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            const response = await fetch('/api/settings');
            if (!response.ok) {
                throw new Error(`Failed to fetch settings: ${response.statusText}`);
            }

            const flatSettings = await response.json();
            this.settings = this.unflattenSettings(flatSettings);
            this.initialized = true;
            logger.info('SettingsStore initialized');

            if (this.options.autoSave) {
                this.syncTimer = window.setInterval(
                    () => this.syncPendingChanges(),
                    this.options.syncInterval
                ) as unknown as number;
            }
        } catch (error) {
            logger.error('Failed to initialize SettingsStore:', error);
            throw error;
        }
    }

    private async syncPendingChanges(): Promise<void> {
        if (!this.initialized || this.pendingChanges.size === 0) {
            return;
        }

        const updates = Array.from(this.pendingChanges).map(async (path) => {
            const value = this.get(path);
            const url = `/api/settings/${path}`;

            try {
                const response = await fetch(url, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(value)
                });

                if (!response.ok) {
                    throw new Error(`Failed to update setting ${path}: ${response.statusText}`);
                }

                return path;
            } catch (error) {
                logger.error(`Failed to sync setting ${path}:`, error);
                throw error;
            }
        });

        try {
            await Promise.all(updates);
            this.pendingChanges.clear();
            logger.info('Settings synced successfully');
        } catch (error) {
            logger.error('Failed to sync some settings:', error);
            throw error;
        }
    }

    public get(path: string): unknown {
        if (!path) {
            return this.settings;
        }
        return path.split('.').reduce((obj: any, key) => obj && obj[key], this.settings);
    }

    public set(path: string, value: unknown): void {
        const parts = path.split('.');
        const lastKey = parts.pop()!;
        const target = parts.reduce((obj: any, key) => {
            if (!(key in obj)) {
                obj[key] = {};
            }
            return obj[key];
        }, this.settings);

        if (target) {
            target[lastKey] = value;
            this.pendingChanges.add(path);
            this.notifySubscribers(path, value);
        }
    }

    private unflattenSettings(flatSettings: Record<string, unknown>): Settings {
        const result: any = {};
        
        for (const [path, value] of Object.entries(flatSettings)) {
            const parts = path.split('.');
            let current = result;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) {
                    current[part] = {};
                }
                current = current[part];
            }
            
            current[parts[parts.length - 1]] = value;
        }
        
        return result as Settings;
    }

    public subscribe(path: string, callback: SettingsChangeCallback): () => void {
        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        const pathSubscribers = this.subscribers.get(path)!;
        pathSubscribers.add(callback);

        return () => {
            const pathSubscribers = this.subscribers.get(path);
            if (pathSubscribers) {
                pathSubscribers.delete(callback);
                if (pathSubscribers.size === 0) {
                    this.subscribers.delete(path);
                }
            }
        };
    }

    private notifySubscribers(path: string, value: unknown): void {
        const subscribers = this.subscribers.get(path);
        if (subscribers) {
            subscribers.forEach(callback => {
                try {
                    callback(path, value);
                } catch (error) {
                    logger.error(`Error in settings subscriber for ${path}:`, error);
                }
            });
        }
    }

    public dispose(): void {
        if (this.syncTimer !== null) {
            window.clearInterval(this.syncTimer);
            this.syncTimer = null;
        }
        this.subscribers.clear();
        this.pendingChanges.clear();
        this.initialized = false;
    }
}
