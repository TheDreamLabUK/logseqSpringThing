// Basic setting value types
export type BasicSettingValue = string | number | boolean;

export interface NodeSettings {
    baseSize: number;
    baseColor: string;
    opacity: number;
    shape: string;
    clearcoat: number;
    enableHoverEffect: boolean;
    highlightColor: string;
    highlightScale: number;
    outlineWidth: number;
    outlineColor: string;
    metalness: number;
    roughness: number;
}

export interface EdgeSettings {
    baseWidth: number;
    baseColor: string;
    opacity: number;
    arrowScale: number;
    enableArrows: boolean;
}

export interface RenderingSettings {
    bloomEnabled: boolean;
    bloomStrength: number;
    bloomThreshold: number;
    bloomRadius: number;
    fov: number;
    near: number;
    far: number;
}

export interface PhysicsSettings {
    gravity: number;
    friction: number;
    springStrength: number;
    springLength: number;
    damping: number;
    attractionStrength: number;
    repulsionStrength: number;
    enabled: boolean;
}

export interface LabelSettings {
    desktopFontSize: number;
    textColor: string;
    offset: number;
    maxVisible: number;
    minScale: number;
    maxScale: number;
    enableLabels: boolean;
}

export interface BloomSettings {
    edgeBloomStrength: number;
}

export interface ClientDebugSettings {
    enabled: boolean;
    showFPS: boolean;
    showStats: boolean;
}

export interface Settings {
    nodes: NodeSettings;
    edges: EdgeSettings;
    rendering: RenderingSettings;
    physics: PhysicsSettings;
    labels: LabelSettings;
    bloom: BloomSettings;
    clientDebug: ClientDebugSettings;
}

export type SettingsCategory = keyof Settings;
export type SettingKey<T extends SettingsCategory> = keyof Settings[T];

// Use this type to get the actual value type for a given setting
export type SettingValueType<T extends SettingsCategory, K extends SettingKey<T>> = Settings[T][K];

export interface SettingsManager {
    getCurrentSettings(): Settings;
    getDefaultSettings(): Settings;
    initialize(): Promise<void>;
    updateSetting<T extends SettingsCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>
    ): Promise<void>;
    subscribe<T extends SettingsCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        callback: (value: SettingValueType<T, K>) => void
    ): () => void;
    dispose(): void;
}
