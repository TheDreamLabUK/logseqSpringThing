import { createLogger } from '../utils/logger';

const logger = createLogger('SettingsManager');

// Server-side settings interface matching settings.toml structure
interface ServerSettings {
    nodes: {
        base_size: number;
        base_color: string;
        opacity: number;
        metalness: number;
        roughness: number;
        clearcoat: number;
        enable_instancing: boolean;
        material_type: string;
        size_range: number[];
        size_by_connections: boolean;
        highlight_color: string;
        highlight_duration: number;
        enable_hover_effect: boolean;
        hover_scale: number;
    };
    edges: {
        arrow_size: number;
        base_width: number;
        color: string;
        enable_arrows: boolean;
        opacity: number;
        width_range: number[];
    };
    physics: {
        attraction_strength: number;
        bounds_size: number;
        collision_radius: number;
        damping: number;
        enable_bounds: boolean;
        enabled: boolean;
        iterations: number;
        max_velocity: number;
        repulsion_strength: number;
        spring_strength: number;
    };
    rendering: {
        ambient_light_intensity: number;
        background_color: string;
        directional_light_intensity: number;
        enable_ambient_occlusion: boolean;
        enable_antialiasing: boolean;
        enable_shadows: boolean;
        environment_intensity: number;
    };
    bloom: {
        edge_bloom_strength: number;
        enabled: boolean;
        environment_bloom_strength: number;
        node_bloom_strength: number;
        radius: number;
        strength: number;
    };
    animations: {
        enable_motion_blur: boolean;
        enable_node_animations: boolean;
        motion_blur_strength: number;
        selection_wave_enabled: boolean;
        pulse_enabled: boolean;
        ripple_enabled: boolean;
        edge_animation_enabled: boolean;
        flow_particles_enabled: boolean;
    };
    labels: {
        desktop_font_size: number;
        enable_labels: boolean;
        text_color: string;
    };
    ar: {
        drag_threshold: number;
        enable_hand_tracking: boolean;
        enable_haptics: boolean;
        enable_light_estimation: boolean;
        enable_passthrough_portal: boolean;
        enable_plane_detection: boolean;
        enable_scene_understanding: boolean;
        gesture_smoothing: number;
        hand_mesh_color: string;
        hand_mesh_enabled: boolean;
        hand_mesh_opacity: number;
        hand_point_size: number;
        hand_ray_color: string;
        hand_ray_enabled: boolean;
        hand_ray_width: number;
        haptic_intensity: number;
        passthrough_brightness: number;
        passthrough_contrast: number;
        passthrough_opacity: number;
        pinch_threshold: number;
        plane_color: string;
        plane_opacity: number;
        portal_edge_color: string;
        portal_edge_width: number;
        portal_size: number;
        room_scale: boolean;
        rotation_threshold: number;
        show_plane_overlay: boolean;
        snap_to_floor: boolean;
    };
}

export class SettingsManager {
    private settings: ServerSettings;
    private subscribers: Map<string, Map<string, Set<(value: any) => void>>> = new Map();
    private connectionSubscribers: Set<(connected: boolean) => void> = new Set();
    private connected: boolean = false;

    constructor() {
        this.settings = this.getDefaultSettings();
        this.initializeSettings();
    }

    private async initializeSettings() {
        try {
            await this.loadAllSettings();
            this.setConnectionStatus(true);
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            this.setConnectionStatus(false);
        }
    }

    private getDefaultSettings(): ServerSettings {
        return {
            nodes: {
                base_size: 1,
                base_color: '#c3ab6f',
                opacity: 0.4,
                metalness: 0.3,
                roughness: 0.35,
                clearcoat: 1,
                enable_instancing: false,
                material_type: 'basic',
                size_range: [1, 5],
                size_by_connections: true,
                highlight_color: '#822626',
                highlight_duration: 300,
                enable_hover_effect: false,
                hover_scale: 1.2
            },
            edges: {
                arrow_size: 0.15,
                base_width: 2,
                color: '#917f18',
                enable_arrows: false,
                opacity: 0.6,
                width_range: [1, 3]
            },
            physics: {
                attraction_strength: 0.015,
                bounds_size: 12,
                collision_radius: 0.25,
                damping: 0.88,
                enable_bounds: true,
                enabled: false,
                iterations: 500,
                max_velocity: 2.5,
                repulsion_strength: 1500,
                spring_strength: 0.018
            },
            rendering: {
                ambient_light_intensity: 0.7,
                background_color: '#000000',
                directional_light_intensity: 1,
                enable_ambient_occlusion: false,
                enable_antialiasing: true,
                enable_shadows: false,
                environment_intensity: 1.2
            },
            bloom: {
                edge_bloom_strength: 0.3,
                enabled: false,
                environment_bloom_strength: 0.5,
                node_bloom_strength: 0.2,
                radius: 0.5,
                strength: 1.8
            },
            animations: {
                enable_motion_blur: false,
                enable_node_animations: false,
                motion_blur_strength: 0.4,
                selection_wave_enabled: false,
                pulse_enabled: false,
                ripple_enabled: false,
                edge_animation_enabled: false,
                flow_particles_enabled: false
            },
            labels: {
                desktop_font_size: 48,
                enable_labels: true,
                text_color: '#FFFFFF'
            },
            ar: {
                drag_threshold: 0.04,
                enable_hand_tracking: true,
                enable_haptics: true,
                enable_light_estimation: true,
                enable_passthrough_portal: false,
                enable_plane_detection: true,
                enable_scene_understanding: true,
                gesture_smoothing: 0.9,
                hand_mesh_color: '#FFD700',
                hand_mesh_enabled: true,
                hand_mesh_opacity: 0.3,
                hand_point_size: 0.01,
                hand_ray_color: '#FFD700',
                hand_ray_enabled: true,
                hand_ray_width: 0.002,
                haptic_intensity: 0.7,
                passthrough_brightness: 1,
                passthrough_contrast: 1,
                passthrough_opacity: 1,
                pinch_threshold: 0.015,
                plane_color: '#4A90E2',
                plane_opacity: 0.3,
                portal_edge_color: '#FFD700',
                portal_edge_width: 0.02,
                portal_size: 1,
                room_scale: true,
                rotation_threshold: 0.08,
                show_plane_overlay: true,
                snap_to_floor: true
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
        
        // Immediately notify with current value
        const currentValue = (this.settings[category as keyof ServerSettings] as any)[setting];
        if (currentValue !== undefined) {
            listener(currentValue);
        }
        
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

        settingSet.forEach(listener => {
            try {
                listener(value);
            } catch (error) {
                logger.error(`Error in settings listener for ${category}.${setting}:`, error);
            }
        });
    }

    private async updateSetting(category: keyof ServerSettings, setting: string, value: any): Promise<void> {
        try {
            const response = await fetch(`/api/visualization/${category}/${setting}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ value })
            });

            if (!response.ok) {
                throw new Error(`Failed to update setting: ${response.statusText}`);
            }

            const result = await response.json();
            const newValue = result.value;

            // Update local state
            (this.settings[category] as any)[setting] = newValue;
            this.notifySubscribers(category, setting, newValue);
            
            logger.info(`Updated ${String(category)}.${setting} to:`, newValue);
        } catch (error) {
            logger.error(`Failed to update ${String(category)}.${setting}:`, error);
            throw error;
        }
    }

    private async getSetting(category: keyof ServerSettings, setting: string): Promise<any> {
        try {
            const response = await fetch(`/api/visualization/${category}/${setting}`);
            if (!response.ok) {
                throw new Error(`Failed to get setting: ${response.statusText}`);
            }
            const data = await response.json();
            return data.value;
        } catch (error) {
            logger.error(`Failed to get ${String(category)}.${setting}:`, error);
            throw error;
        }
    }

    public async loadAllSettings(): Promise<void> {
        const categories = Object.keys(this.settings) as Array<keyof ServerSettings>;
        
        for (const category of categories) {
            const settings = Object.keys(this.settings[category]);
            for (const setting of settings) {
                try {
                    const value = await this.getSetting(category, setting);
                    (this.settings[category] as any)[setting] = value;
                    this.notifySubscribers(category, setting, value);
                } catch (error) {
                    logger.error(`Failed to load ${String(category)}.${setting}:`, error);
                }
            }
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
        this.connectionSubscribers.forEach(listener => {
            try {
                listener(connected);
            } catch (error) {
                logger.error('Error in connection status listener:', error);
            }
        });
    }

    public getCurrentSettings(): ServerSettings {
        return JSON.parse(JSON.stringify(this.settings));
    }

    public dispose(): void {
        this.subscribers.clear();
        this.connectionSubscribers.clear();
    }
}

// Create singleton instance
export const settingsManager = new SettingsManager();

// Re-export Settings interface
export type Settings = ServerSettings;
