import { Settings } from '../types/settings/base';
import { defaultSettings } from '../state/defaultSettings';
import { createLogger } from '../core/logger';
import { validateSettings } from '../types/settings/validation';

const logger = createLogger('SettingsPersistenceService');

export interface StoredSettings {
    settings: Settings;
    timestamp: number;
    version: string;
    pubkey?: string;
}

export class SettingsPersistenceService {
    private static instance: SettingsPersistenceService | null = null;
    private readonly LOCAL_STORAGE_KEY = 'logseq_spring_settings';
    private readonly SETTINGS_VERSION = '1.0.0';
    private currentPubkey: string | null = null;

    private constructor() {}

    public static getInstance(): SettingsPersistenceService {
        if (!SettingsPersistenceService.instance) {
            SettingsPersistenceService.instance = new SettingsPersistenceService();
        }
        return SettingsPersistenceService.instance;
    }

    public setCurrentPubkey(pubkey: string | null): void {
        this.currentPubkey = pubkey;
    }

    public async saveSettings(settings: Settings): Promise<void> {
        try {
            // Validate settings before saving
            const validation = validateSettings(settings);
            if (!validation.isValid) {
                throw new Error(`Invalid settings: ${JSON.stringify(validation.errors)}`);
            }

            const storedSettings: StoredSettings = {
                settings,
                timestamp: Date.now(),
                version: this.SETTINGS_VERSION,
                pubkey: this.currentPubkey ?? undefined
            };

            // Save locally
            localStorage.setItem(this.LOCAL_STORAGE_KEY, JSON.stringify(storedSettings));

            // If user is authenticated, sync to server
            if (this.currentPubkey) {
                await this.syncToServer(storedSettings);
            }

            logger.info('Settings saved successfully');
        } catch (error) {
            logger.error('Failed to save settings:', error);
            throw error;
        }
    }

    public async loadSettings(): Promise<Settings> {
        try {
            // Try to load from server if authenticated
            if (this.currentPubkey) {
                try {
                    const serverSettings = await this.loadFromServer();
                    if (serverSettings) {
                        return serverSettings;
                    }
                } catch (error) {
                    logger.warn('Failed to load settings from server:', error);
                }
            }

            // Fall back to local storage
            const storedJson = localStorage.getItem(this.LOCAL_STORAGE_KEY);
            if (storedJson) {
                const stored: StoredSettings = JSON.parse(storedJson);

                // Version check
                if (stored.version !== this.SETTINGS_VERSION) {
                    logger.warn('Settings version mismatch, using defaults');
                    return this.migrateSettings(stored.settings);
                }

                // Pubkey check
                if (stored.pubkey && stored.pubkey !== this.currentPubkey) {
                    logger.warn('Settings pubkey mismatch, using defaults');
                    return { ...defaultSettings };
                }

                // Validate loaded settings
                const validation = validateSettings(stored.settings);
                if (!validation.isValid) {
                    logger.warn('Invalid stored settings, using defaults');
                    return { ...defaultSettings };
                }

                return stored.settings;
            }

            // No stored settings found, use defaults
            return { ...defaultSettings };
        } catch (error) {
            logger.error('Failed to load settings:', error);
            return { ...defaultSettings };
        }
    }

    private async syncToServer(storedSettings: StoredSettings): Promise<void> {
        try {
            const response = await fetch('/api/settings/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Nostr-Pubkey': this.currentPubkey!
                },
                body: JSON.stringify(storedSettings)
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${await response.text()}`);
            }

            logger.info('Settings synced to server');
        } catch (error) {
            logger.error('Failed to sync settings to server:', error);
            throw error;
        }
    }

    private async loadFromServer(): Promise<Settings | null> {
        try {
            const response = await fetch('/api/settings/sync', {
                headers: {
                    'X-Nostr-Pubkey': this.currentPubkey!
                }
            });

            if (!response.ok) {
                if (response.status === 404) {
                    return null;
                }
                throw new Error(`Server returned ${response.status}: ${await response.text()}`);
            }

            const stored: StoredSettings = await response.json();
            
            // Version check
            if (stored.version !== this.SETTINGS_VERSION) {
                return this.migrateSettings(stored.settings);
            }

            return stored.settings;
        } catch (error) {
            logger.error('Failed to load settings from server:', error);
            throw error;
        }
    }

    private migrateSettings(oldSettings: Settings): Settings {
        // Implement version-specific migrations here
        logger.info('Migrating settings from older version');
        
        // For now, just merge with defaults
        return {
            ...defaultSettings,
            ...oldSettings,
            // Ensure critical sections are preserved
            system: {
                ...defaultSettings.system,
                ...oldSettings.system
            },
            xr: {
                ...defaultSettings.xr,
                ...oldSettings.xr
            }
        };
    }

    public clearSettings(): void {
        localStorage.removeItem(this.LOCAL_STORAGE_KEY);
        logger.info('Settings cleared');
    }

    public dispose(): void {
        SettingsPersistenceService.instance = null;
    }
}