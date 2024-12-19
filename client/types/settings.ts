import { Settings, SettingCategory, SettingKey } from '../core/types';
export type { Settings, SettingCategory, SettingKey };

// Helper type to get the value type for a specific setting
export type SettingValueType<T extends SettingCategory, K extends SettingKey<T>> = Settings[T][K];

export interface SettingsManager {
    getCurrentSettings(): Settings;
    getDefaultSettings(): Settings;
    initialize(): Promise<void>;
    updateSetting<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>
    ): Promise<void>;
    subscribe<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        callback: (value: SettingValueType<T, K>) => void
    ): () => void;
    dispose(): void;
}
