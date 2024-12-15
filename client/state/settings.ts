import { createLogger } from '../utils/logger';

const logger = createLogger('SettingsManager');

export interface NodeSettings {
    size: number;
    color: string;
    opacity: number;
    shape: 'sphere' | 'cube' | 'cylinder' | 'cone';
    texture: 'smooth' | 'rough' | 'metallic';
    outlineWidth: number;
    outlineColor: string;
    glowIntensity: number;
    glowColor: string;
    highlightColor: string;
}

export interface EdgeSettings {
    width: number;
    color: string;
    opacity: number;
    style: 'solid' | 'dashed' | 'dotted';
    arrowSize: number;
    arrowPosition: 'none' | 'forward' | 'backward' | 'both';
}

export interface PhysicsSettings {
    enabled: boolean;
    gravity: number;
    springLength: number;
    springStrength: number;
    repulsion: number;
    damping: number;
}

export interface RenderingSettings {
    antialias: boolean;
    shadows: boolean;
    ambientLight: number;
    directionalLight: number;
    background: string;
}

export interface BloomSettings {
    enabled: boolean;
    strength: number;
    radius: number;
    threshold: number;
}

export interface AnimationSettings {
    enabled: boolean;
    duration: number;
    easing: 'linear' | 'easeIn' | 'easeOut' | 'easeInOut';
}

export interface LabelSettings {
    enabled: boolean;
    size: number;
    color: string;
    background: string;
    offset: number;
}

export interface ARSettings {
    enabled: boolean;
    markerType: 'pattern' | 'barcode';
    markerSize: number;
    autoRotate: boolean;
}

export interface VisualizationSettings {
    nodes: NodeSettings;
    edges: EdgeSettings;
    physics: PhysicsSettings;
    rendering: RenderingSettings;
    bloom: BloomSettings;
    animation: AnimationSettings;
    label: LabelSettings;
    ar: ARSettings;
}

export class SettingsManager {
    private settings: VisualizationSettings;
    private subscribers: Map<string, Map<string, Set<(value: any) => void>>> = new Map();
    private connectionSubscribers: Set<(connected: boolean) => void> = new Set();
    private connected: boolean = false;

    constructor() {
        this.settings = this.getDefaultSettings();
    }

    private getDefaultSettings(): VisualizationSettings {
        return {
            nodes: {
                size: 1,
                color: '#1E90FF',
                opacity: 1,
                shape: 'sphere',
                texture: 'smooth',
                outlineWidth: 0,
                outlineColor: '#000000',
                glowIntensity: 0,
                glowColor: '#FFFFFF',
                highlightColor: '#FF4500'
            },
            edges: {
                width: 1,
                color: '#808080',
                opacity: 1,
                style: 'solid',
                arrowSize: 1,
                arrowPosition: 'none'
            },
            physics: {
                enabled: true,
                gravity: 0,
                springLength: 100,
                springStrength: 0.1,
                repulsion: 100,
                damping: 0.1
            },
            rendering: {
                antialias: true,
                shadows: true,
                ambientLight: 0.5,
                directionalLight: 0.8,
                background: '#FFFFFF'
            },
            bloom: {
                enabled: false,
                strength: 1,
                radius: 1,
                threshold: 0.5
            },
            animation: {
                enabled: true,
                duration: 500,
                easing: 'easeInOut'
            },
            label: {
                enabled: true,
                size: 12,
                color: '#000000',
                background: '#FFFFFF',
                offset: 5
            },
            ar: {
                enabled: false,
                markerType: 'pattern',
                markerSize: 1,
                autoRotate: false
            }
        };
    }

    public subscribe<T>(category: string, setting: string, listener: (value: T) => void): () => void {
        if (!this.subscribers.has(category)) {
            this.subscribers.set(category, new Map());
        }
        const categoryMap = this.subscribers.get(category)!;
        
        if (!categoryMap.has(setting)) {
            categoryMap.set(setting, new Set());
        }
        const settingSet = categoryMap.get(setting)!;
        
        settingSet.add(listener);
        
        return () => {
            settingSet.delete(listener);
            if (settingSet.size === 0) {
                categoryMap.delete(setting);
            }
            if (categoryMap.size === 0) {
                this.subscribers.delete(category);
            }
        };
    }

    private notifySubscribers<T>(category: string, setting: string, value: T): void {
        const categoryMap = this.subscribers.get(category);
        if (!categoryMap) return;

        const settingSet = categoryMap.get(setting);
        if (!settingSet) return;

        settingSet.forEach(listener => listener(value));
    }

    public async updateNodeSetting<T extends keyof NodeSettings>(setting: T, value: NodeSettings[T]): Promise<void> {
        this.settings.nodes[setting] = value;
        this.notifySubscribers('nodes', setting as string, value);
    }

    public async updateEdgeSetting<T extends keyof EdgeSettings>(setting: T, value: EdgeSettings[T]): Promise<void> {
        this.settings.edges[setting] = value;
        this.notifySubscribers('edges', setting as string, value);
    }

    public async updatePhysicsSetting<T extends keyof PhysicsSettings>(setting: T, value: PhysicsSettings[T]): Promise<void> {
        this.settings.physics[setting] = value;
        this.notifySubscribers('physics', setting as string, value);
    }

    public async updateRenderingSetting<T extends keyof RenderingSettings>(setting: T, value: RenderingSettings[T]): Promise<void> {
        this.settings.rendering[setting] = value;
        this.notifySubscribers('rendering', setting as string, value);
    }

    public async updateBloomSetting<T extends keyof BloomSettings>(setting: T, value: BloomSettings[T]): Promise<void> {
        this.settings.bloom[setting] = value;
        this.notifySubscribers('bloom', setting as string, value);
    }

    public async updateAnimationSetting<T extends keyof AnimationSettings>(setting: T, value: AnimationSettings[T]): Promise<void> {
        this.settings.animation[setting] = value;
        this.notifySubscribers('animation', setting as string, value);
    }

    public async updateLabelSetting<T extends keyof LabelSettings>(setting: T, value: LabelSettings[T]): Promise<void> {
        this.settings.label[setting] = value;
        this.notifySubscribers('label', setting as string, value);
    }

    public async updateARSetting<T extends keyof ARSettings>(setting: T, value: ARSettings[T]): Promise<void> {
        this.settings.ar[setting] = value;
        this.notifySubscribers('ar', setting as string, value);
    }

    public async getNodeSetting<T extends keyof NodeSettings>(setting: T): Promise<NodeSettings[T]> {
        return this.settings.nodes[setting];
    }

    public async getEdgeSetting<T extends keyof EdgeSettings>(setting: T): Promise<EdgeSettings[T]> {
        return this.settings.edges[setting];
    }

    public async getPhysicsSetting<T extends keyof PhysicsSettings>(setting: T): Promise<PhysicsSettings[T]> {
        return this.settings.physics[setting];
    }

    public async getRenderingSetting<T extends keyof RenderingSettings>(setting: T): Promise<RenderingSettings[T]> {
        return this.settings.rendering[setting];
    }

    public async getBloomSetting<T extends keyof BloomSettings>(setting: T): Promise<BloomSettings[T]> {
        return this.settings.bloom[setting];
    }

    public async getAnimationSetting<T extends keyof AnimationSettings>(setting: T): Promise<AnimationSettings[T]> {
        return this.settings.animation[setting];
    }

    public async getLabelSetting<T extends keyof LabelSettings>(setting: T): Promise<LabelSettings[T]> {
        return this.settings.label[setting];
    }

    public async getARSetting<T extends keyof ARSettings>(setting: T): Promise<ARSettings[T]> {
        return this.settings.ar[setting];
    }

    public async updateSettings(settings: Partial<VisualizationSettings>): Promise<void> {
        Object.entries(settings).forEach(([category, value]) => {
            const key = category as keyof VisualizationSettings;
            if (key in this.settings) {
                // Type assertion is safe here because we checked that the key exists
                const typedValue = value as VisualizationSettings[typeof key];
                Object.assign(this.settings[key], typedValue);
                this.notifySubscribers(category, '', typedValue);
            }
        });
        await this.saveSettings();
    }

    public async saveSettings(): Promise<void> {
        try {
            // Save settings to backend
            const response = await fetch('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.settings)
            });
            if (!response.ok) throw new Error('Failed to save settings');
            logger.info('Settings saved successfully');
        } catch (error) {
            logger.error('Failed to save settings:', error);
            throw error;
        }
    }

    public subscribeToConnection(listener: (connected: boolean) => void): () => void {
        this.connectionSubscribers.add(listener);
        listener(this.connected);
        return () => {
            this.connectionSubscribers.delete(listener);
        };
    }

    public setConnectionStatus(connected: boolean): void {
        this.connected = connected;
        this.connectionSubscribers.forEach(listener => listener(connected));
    }

    public getCurrentSettings(): VisualizationSettings {
        return { ...this.settings };
    }

    public getThreeJSSettings(): VisualizationSettings {
        return this.getCurrentSettings();
    }

    public dispose(): void {
        this.subscribers.clear();
    }
}

// Create singleton instance
export const settingsManager = new SettingsManager();

// Re-export Settings interface
export type Settings = VisualizationSettings;
