import { Logger } from '../utils/logger';
import { defaultSettings } from './defaultSettings';

const log = new Logger('SettingsStore');

export interface Setting {
    value: any;
    type: string;
    description?: string;
    min?: number;
    max?: number;
    options?: string[];
}

export class SettingsStore {
    private settings: Map<string, Setting> = new Map();
    private subscribers: Map<string, Set<(value: any) => void>> = new Map();

    public async initialize(): Promise<void> {
        try {
            const response = await fetch('/api/visualization/settings');
            if (!response.ok) {
                throw new Error(`Failed to load settings: ${response.statusText}`);
            }
            const settings = await response.json();
            this.settings = new Map(Object.entries(settings));
        } catch (error) {
            log.error('Failed to load settings:', error);
            // Use default settings
            this.settings = new Map(Object.entries(defaultSettings));
        }
    }

    public get(path: string): Setting | undefined {
        return this.settings.get(path);
    }

    public set(path: string, value: any): void {
        const setting = this.settings.get(path);
        if (!setting) {
            throw new Error(`Setting not found: ${path}`);
        }

        // Validate value based on setting type
        this.validateValue(setting, value);

        setting.value = value;
        this.notifySubscribers(path, value);
    }

    private validateValue(setting: Setting, value: any): void {
        switch (setting.type) {
            case 'number':
                if (typeof value !== 'number') {
                    throw new Error('Value must be a number');
                }
                if (setting.min !== undefined && value < setting.min) {
                    throw new Error(`Value must be >= ${setting.min}`);
                }
                if (setting.max !== undefined && value > setting.max) {
                    throw new Error(`Value must be <= ${setting.max}`);
                }
                break;
            case 'string':
                if (typeof value !== 'string') {
                    throw new Error('Value must be a string');
                }
                if (setting.options && !setting.options.includes(value)) {
                    throw new Error(`Value must be one of: ${setting.options.join(', ')}`);
                }
                break;
            case 'boolean':
                if (typeof value !== 'boolean') {
                    throw new Error('Value must be a boolean');
                }
                break;
        }
    }

    private notifySubscribers(path: string, value: any): void {
        const subscribers = this.subscribers.get(path);
        if (subscribers) {
            subscribers.forEach(callback => callback(value));
        }
    }

    public subscribe(path: string, callback: (value: any) => void): () => void {
        if (!this.validatePath(path)) {
            throw new Error(`Invalid setting path: ${path}`);
        }

        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path)!.add(callback);

        return () => {
            this.subscribers.get(path)?.delete(callback);
        };
    }

    private validatePath(path: string): boolean {
        return /^[a-z0-9_\-/.]+$/i.test(path);
    }

    public dispose(): void {
        this.settings.clear();
        this.subscribers.clear();
    }

    public async batchUpdate(updates: Array<{path: string, value: any}>): Promise<void> {
        const backup = new Map(this.settings);
        const errors: Error[] = [];

        try {
            for (const {path, value} of updates) {
                try {
                    await this.set(path, value);
                } catch (error) {
                    errors.push(error as Error);
                    break;
                }
            }

            if (errors.length > 0) {
                // Roll back all changes
                this.settings = backup;
                throw new Error(`Batch update failed: ${errors[0].message}`);
            }
        } catch (error) {
            // Restore original state
            this.settings = backup;
            throw error;
        }
    }

    public getCategory(category: string): Map<string, Setting> {
        // Return a new map with only settings in the specified category
        return new Map(
            Array.from(this.settings.entries())
                .filter(([path]) => path.startsWith(category))
        );
    }

    public async retryInitialize(maxRetries: number = 3): Promise<void> {
        let lastError: Error | null = null;
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                await this.initialize();
                return;
            } catch (error) {
                lastError = error as Error;
                log.warn(`Initialize attempt ${i + 1} failed:`, error);
                await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
            }
        }

        log.error('All initialize attempts failed, using default settings');
        this.settings = new Map(Object.entries(defaultSettings));
        
        if (lastError) {
            throw lastError;
        }
    }
} 