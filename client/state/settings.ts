/**
 * Settings management with simplified visualization configuration
 */

import { VisualizationSettings, ServerSettings } from '../core/types';
import { createLogger } from '../core/utils';

const logger = createLogger('SettingsManager');

// Default settings matching settings.toml exactly
export const DEFAULT_VISUALIZATION_SETTINGS: VisualizationSettings = {
    // Node Appearance
    nodeSize: 1.0,  // maps from nodes.base_size
    nodeColor: '#c3ab6f',  // maps from nodes.base_color
    nodeOpacity: 0.4,  // maps from nodes.opacity
    metalness: 0.3,  // maps from nodes.metalness
    roughness: 0.35,  // maps from nodes.roughness
    clearcoat: 1.0,  // maps from nodes.clearcoat
    enableInstancing: false,  // maps from nodes.enable_instancing
    materialType: 'basic',  // maps from nodes.material_type
    sizeRange: [1, 5],  // maps from nodes.size_range
    sizeByConnections: true,  // maps from nodes.size_by_connections
    highlightColor: '#822626',  // maps from nodes.highlight_color
    highlightDuration: 300,  // maps from nodes.highlight_duration
    enableHoverEffect: false,  // maps from nodes.enable_hover_effect
    hoverScale: 1.2,  // maps from nodes.hover_scale

    // Edge Appearance
    edgeWidth: 2.0,  // maps from edges.base_width
    edgeColor: '#917f18',  // maps from edges.color
    edgeOpacity: 0.6,  // maps from edges.opacity
    edgeWidthRange: [1, 3],  // maps from edges.width_range
    enableArrows: false,  // maps from edges.enable_arrows
    arrowSize: 0.15,  // maps from edges.arrow_size

    // Physics Settings
    physicsEnabled: false,  // maps from physics.enabled
    attractionStrength: 0.015,  // maps from physics.attraction_strength
    repulsionStrength: 1500.0,  // maps from physics.repulsion_strength
    springStrength: 0.018,  // maps from physics.spring_strength
    damping: 0.88,  // maps from physics.damping
    maxVelocity: 2.5,  // maps from physics.max_velocity
    collisionRadius: 0.25,  // maps from physics.collision_radius
    boundsSize: 12.0,  // maps from physics.bounds_size
    enableBounds: true,  // maps from physics.enable_bounds
    iterations: 500,  // maps from physics.iterations

    // Rendering Settings
    ambientLightIntensity: 0.7,  // maps from rendering.ambient_light_intensity
    directionalLightIntensity: 1.0,  // maps from rendering.directional_light_intensity
    environmentIntensity: 1.2,  // maps from rendering.environment_intensity
    enableAmbientOcclusion: false,  // maps from rendering.enable_ambient_occlusion
    enableAntialiasing: true,  // maps from rendering.enable_antialiasing
    enableShadows: false,  // maps from rendering.enable_shadows
    backgroundColor: '#000000',  // maps from rendering.background_color

    // Visual Effects
    enableBloom: false,  // maps from bloom.enabled
    bloomIntensity: 1.8,  // maps from bloom.strength
    bloomRadius: 0.5,  // maps from bloom.radius
    nodeBloomStrength: 0.2,  // maps from bloom.node_bloom_strength
    edgeBloomStrength: 0.3,  // maps from bloom.edge_bloom_strength
    environmentBloomStrength: 0.5,  // maps from bloom.environment_bloom_strength
    enableNodeAnimations: false,  // maps from animations.enable_node_animations
    enableMotionBlur: false,  // maps from animations.enable_motion_blur
    motionBlurStrength: 0.4,  // maps from animations.motion_blur_strength

    // Labels
    showLabels: true,  // maps from labels.enable_labels
    labelSize: 1.0,  // maps from labels.desktop_font_size / 48
    labelColor: '#FFFFFF',  // maps from labels.text_color

    // Performance
    maxFps: 60,  // Not in server settings

    // AR Settings
    enablePlaneDetection: true,  // maps from ar.enable_plane_detection
    enableSceneUnderstanding: true,  // maps from ar.enable_scene_understanding
    showPlaneOverlay: true,  // maps from ar.show_plane_overlay
    planeOpacity: 0.3,  // maps from ar.plane_opacity
    planeColor: '#4A90E2',  // maps from ar.plane_color
    enableLightEstimation: true,  // maps from ar.enable_light_estimation
    enableHandTracking: true,  // maps from ar.enable_hand_tracking
    handMeshEnabled: true,  // maps from ar.hand_mesh_enabled
    handMeshColor: '#FFD700',  // maps from ar.hand_mesh_color
    handMeshOpacity: 0.3,  // maps from ar.hand_mesh_opacity
    handRayEnabled: true,  // maps from ar.hand_ray_enabled
    handRayColor: '#FFD700',  // maps from ar.hand_ray_color
    handRayWidth: 0.002,  // maps from ar.hand_ray_width
    handPointSize: 0.01,  // maps from ar.hand_point_size
    gestureSmoothing: 0.9,  // maps from ar.gesture_smoothing
    pinchThreshold: 0.015,  // maps from ar.pinch_threshold
    dragThreshold: 0.04,  // maps from ar.drag_threshold
    rotationThreshold: 0.08,  // maps from ar.rotation_threshold
    enableHaptics: true,  // maps from ar.enable_haptics
    hapticIntensity: 0.7,  // maps from ar.haptic_intensity
    roomScale: true,  // maps from ar.room_scale
    snapToFloor: true,  // maps from ar.snap_to_floor
    passthroughOpacity: 1.0,  // maps from ar.passthrough_opacity
    passthroughBrightness: 1.0,  // maps from ar.passthrough_brightness
    passthroughContrast: 1.0,  // maps from ar.passthrough_contrast
    enablePassthroughPortal: false,  // maps from ar.enable_passthrough_portal
    portalSize: 1.0,  // maps from ar.portal_size
    portalEdgeColor: '#FFD700',  // maps from ar.portal_edge_color
    portalEdgeWidth: 0.02  // maps from ar.portal_edge_width
};

// ... [Rest of the file remains the same] ...

export class SettingsManager {
    private static instance: SettingsManager | null = null;
    private settings: VisualizationSettings;
    private settingsListeners: Set<(settings: VisualizationSettings) => void>;
    private connectionListeners: Set<(connected: boolean) => void>;
    private connected: boolean = false;
    private readonly API_BASE = '/api/visualization';

    private constructor() {
        this.settings = { ...DEFAULT_VISUALIZATION_SETTINGS };
        this.settingsListeners = new Set();
        this.connectionListeners = new Set();
        logger.log('Initialized with default settings');
    }

    static getInstance(): SettingsManager {
        if (!SettingsManager.instance) {
            SettingsManager.instance = new SettingsManager();
        }
        return SettingsManager.instance;
    }

    isConnected(): boolean {
        return this.connected;
    }

    onConnectionChange(listener: (connected: boolean) => void): void {
        this.connectionListeners.add(listener);
        // Immediately notify of current state
        listener(this.connected);
    }

    private setConnected(value: boolean): void {
        if (this.connected !== value) {
            this.connected = value;
            this.notifyConnectionListeners();
        }
    }

    private notifyConnectionListeners(): void {
        this.connectionListeners.forEach(listener => {
            try {
                listener(this.connected);
            } catch (error) {
                logger.error('Error in connection listener:', error);
            }
        });
    }

    private notifyListeners(): void {
        this.settingsListeners.forEach(listener => {
            try {
                listener(this.settings);
            } catch (error) {
                logger.error('Error in settings listener:', error);
            }
        });
    }

    async loadSettings(): Promise<void> {
        try {
            const response = await fetch(`${this.API_BASE}/settings`);
            if (!response.ok) {
                this.setConnected(false);
                throw new Error(`Failed to fetch settings: ${response.status} ${response.statusText}`);
            }
            const serverSettings = await response.json();
            this.settings = this.flattenSettings(serverSettings);
            this.notifyListeners();
            this.setConnected(true);
            logger.log('Settings loaded from server:', this.settings);
        } catch (error) {
            logger.error('Failed to load settings:', error);
            this.setConnected(false);
            this.settings = { ...DEFAULT_VISUALIZATION_SETTINGS };
            this.notifyListeners();
        }
    }

    async saveSettings(): Promise<void> {
        try {
            const serverSettings: ServerSettings = {
                nodes: {
                    base_size: this.settings.nodeSize,
                    base_color: this.settings.nodeColor,
                    opacity: this.settings.nodeOpacity,
                    metalness: this.settings.metalness,
                    roughness: this.settings.roughness,
                    clearcoat: this.settings.clearcoat,
                    enable_instancing: this.settings.enableInstancing,
                    material_type: this.settings.materialType,
                    size_range: this.settings.sizeRange,
                    size_by_connections: this.settings.sizeByConnections,
                    highlight_color: this.settings.highlightColor,
                    highlight_duration: this.settings.highlightDuration,
                    enable_hover_effect: this.settings.enableHoverEffect,
                    hover_scale: this.settings.hoverScale
                },
                edges: {
                    base_width: this.settings.edgeWidth,
                    color: this.settings.edgeColor,
                    opacity: this.settings.edgeOpacity,
                    width_range: this.settings.edgeWidthRange,
                    enable_arrows: this.settings.enableArrows,
                    arrow_size: this.settings.arrowSize
                },
                physics: {
                    enabled: this.settings.physicsEnabled,
                    attraction_strength: this.settings.attractionStrength,
                    repulsion_strength: this.settings.repulsionStrength,
                    spring_strength: this.settings.springStrength,
                    damping: this.settings.damping,
                    max_velocity: this.settings.maxVelocity,
                    collision_radius: this.settings.collisionRadius,
                    bounds_size: this.settings.boundsSize,
                    enable_bounds: this.settings.enableBounds,
                    iterations: this.settings.iterations
                },
                rendering: {
                    ambient_light_intensity: this.settings.ambientLightIntensity,
                    directional_light_intensity: this.settings.directionalLightIntensity,
                    environment_intensity: this.settings.environmentIntensity,
                    enable_ambient_occlusion: this.settings.enableAmbientOcclusion,
                    enable_antialiasing: this.settings.enableAntialiasing,
                    enable_shadows: this.settings.enableShadows,
                    background_color: this.settings.backgroundColor
                },
                bloom: {
                    enabled: this.settings.enableBloom,
                    strength: this.settings.bloomIntensity,
                    radius: this.settings.bloomRadius,
                    node_bloom_strength: this.settings.nodeBloomStrength,
                    edge_bloom_strength: this.settings.edgeBloomStrength,
                    environment_bloom_strength: this.settings.environmentBloomStrength
                },
                animations: {
                    enable_node_animations: this.settings.enableNodeAnimations,
                    enable_motion_blur: this.settings.enableMotionBlur,
                    motion_blur_strength: this.settings.motionBlurStrength
                },
                labels: {
                    enable_labels: this.settings.showLabels,
                    desktop_font_size: this.settings.labelSize * 48,
                    text_color: this.settings.labelColor
                },
                ar: {
                    enable_plane_detection: this.settings.enablePlaneDetection,
                    enable_scene_understanding: this.settings.enableSceneUnderstanding,
                    show_plane_overlay: this.settings.showPlaneOverlay,
                    plane_opacity: this.settings.planeOpacity,
                    plane_color: this.settings.planeColor,
                    enable_light_estimation: this.settings.enableLightEstimation,
                    enable_hand_tracking: this.settings.enableHandTracking,
                    hand_mesh_enabled: this.settings.handMeshEnabled,
                    hand_mesh_color: this.settings.handMeshColor,
                    hand_mesh_opacity: this.settings.handMeshOpacity,
                    hand_ray_enabled: this.settings.handRayEnabled,
                    hand_ray_color: this.settings.handRayColor,
                    hand_ray_width: this.settings.handRayWidth,
                    hand_point_size: this.settings.handPointSize,
                    gesture_smoothing: this.settings.gestureSmoothing,
                    pinch_threshold: this.settings.pinchThreshold,
                    drag_threshold: this.settings.dragThreshold,
                    rotation_threshold: this.settings.rotationThreshold,
                    enable_haptics: this.settings.enableHaptics,
                    haptic_intensity: this.settings.hapticIntensity,
                    room_scale: this.settings.roomScale,
                    snap_to_floor: this.settings.snapToFloor,
                    passthrough_opacity: this.settings.passthroughOpacity,
                    passthrough_brightness: this.settings.passthroughBrightness,
                    passthrough_contrast: this.settings.passthroughContrast,
                    enable_passthrough_portal: this.settings.enablePassthroughPortal,
                    portal_size: this.settings.portalSize,
                    portal_edge_color: this.settings.portalEdgeColor,
                    portal_edge_width: this.settings.portalEdgeWidth
                },
                audio: {
                    enable_ambient_sounds: false,
                    enable_interaction_sounds: false,
                    enable_spatial_audio: false
                }
            };

            const response = await fetch(`${this.API_BASE}/settings`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(serverSettings)
            });

            if (!response.ok) {
                this.setConnected(false);
                throw new Error(`Failed to save settings: ${response.status} ${response.statusText}`);
            }

            const updatedSettings = await response.json();
            this.settings = this.flattenSettings(updatedSettings);
            this.notifyListeners();
            this.setConnected(true);
            logger.log('Settings saved successfully');
        } catch (error) {
            this.setConnected(false);
            logger.error('Failed to save settings:', error);
            throw error;
        }
    }

    updateSettings(newSettings: Partial<VisualizationSettings>): void {
        this.settings = { ...this.settings, ...newSettings };
        logger.log('Updated settings locally');
        this.notifyListeners();
        this.saveSettings().catch(error => {
            logger.error('Failed to save settings to server:', error);
        });
    }

    private flattenSettings(serverSettings: ServerSettings): VisualizationSettings {
        return {
            // Node settings
            nodeSize: serverSettings.nodes.base_size,
            nodeColor: serverSettings.nodes.base_color,
            nodeOpacity: serverSettings.nodes.opacity,
            metalness: serverSettings.nodes.metalness,
            roughness: serverSettings.nodes.roughness,
            clearcoat: serverSettings.nodes.clearcoat,
            enableInstancing: serverSettings.nodes.enable_instancing,
            materialType: serverSettings.nodes.material_type,
            sizeRange: serverSettings.nodes.size_range,
            sizeByConnections: serverSettings.nodes.size_by_connections,
            highlightColor: serverSettings.nodes.highlight_color,
            highlightDuration: serverSettings.nodes.highlight_duration,
            enableHoverEffect: serverSettings.nodes.enable_hover_effect,
            hoverScale: serverSettings.nodes.hover_scale,

            // Edge settings
            edgeWidth: serverSettings.edges.base_width,
            edgeColor: serverSettings.edges.color,
            edgeOpacity: serverSettings.edges.opacity,
            edgeWidthRange: serverSettings.edges.width_range,
            enableArrows: serverSettings.edges.enable_arrows,
            arrowSize: serverSettings.edges.arrow_size,

            // Physics settings
            physicsEnabled: serverSettings.physics.enabled,
            attractionStrength: serverSettings.physics.attraction_strength,
            repulsionStrength: serverSettings.physics.repulsion_strength,
            springStrength: serverSettings.physics.spring_strength,
            damping: serverSettings.physics.damping,
            maxVelocity: serverSettings.physics.max_velocity,
            collisionRadius: serverSettings.physics.collision_radius,
            boundsSize: serverSettings.physics.bounds_size,
            enableBounds: serverSettings.physics.enable_bounds,
            iterations: serverSettings.physics.iterations,

            // Rendering settings
            ambientLightIntensity: serverSettings.rendering.ambient_light_intensity,
            directionalLightIntensity: serverSettings.rendering.directional_light_intensity,
            environmentIntensity: serverSettings.rendering.environment_intensity,
            enableAmbientOcclusion: serverSettings.rendering.enable_ambient_occlusion,
            enableAntialiasing: serverSettings.rendering.enable_antialiasing,
            enableShadows: serverSettings.rendering.enable_shadows,
            backgroundColor: serverSettings.rendering.background_color,

            // Bloom settings
            enableBloom: serverSettings.bloom.enabled,
            bloomIntensity: serverSettings.bloom.strength,
            bloomRadius: serverSettings.bloom.radius,
            nodeBloomStrength: serverSettings.bloom.node_bloom_strength,
            edgeBloomStrength: serverSettings.bloom.edge_bloom_strength,
            environmentBloomStrength: serverSettings.bloom.environment_bloom_strength,

            // Animation settings
            enableNodeAnimations: serverSettings.animations.enable_node_animations,
            enableMotionBlur: serverSettings.animations.enable_motion_blur,
            motionBlurStrength: serverSettings.animations.motion_blur_strength,

            // Label settings
            showLabels: serverSettings.labels.enable_labels,
            labelSize: serverSettings.labels.desktop_font_size / 48,
            labelColor: serverSettings.labels.text_color,

            // Performance settings (not in server settings)
            maxFps: this.settings.maxFps,

            // AR settings
            enablePlaneDetection: serverSettings.ar.enable_plane_detection,
            enableSceneUnderstanding: serverSettings.ar.enable_scene_understanding,
            showPlaneOverlay: serverSettings.ar.show_plane_overlay,
            planeOpacity: serverSettings.ar.plane_opacity,
            planeColor: serverSettings.ar.plane_color,
            enableLightEstimation: serverSettings.ar.enable_light_estimation,
            enableHandTracking: serverSettings.ar.enable_hand_tracking,
            handMeshEnabled: serverSettings.ar.hand_mesh_enabled,
            handMeshColor: serverSettings.ar.hand_mesh_color,
            handMeshOpacity: serverSettings.ar.hand_mesh_opacity,
            handRayEnabled: serverSettings.ar.hand_ray_enabled,
            handRayColor: serverSettings.ar.hand_ray_color,
            handRayWidth: serverSettings.ar.hand_ray_width,
            handPointSize: serverSettings.ar.hand_point_size,
            gestureSmoothing: serverSettings.ar.gesture_smoothing,
            pinchThreshold: serverSettings.ar.pinch_threshold,
            dragThreshold: serverSettings.ar.drag_threshold,
            rotationThreshold: serverSettings.ar.rotation_threshold,
            enableHaptics: serverSettings.ar.enable_haptics,
            hapticIntensity: serverSettings.ar.haptic_intensity,
            roomScale: serverSettings.ar.room_scale,
            snapToFloor: serverSettings.ar.snap_to_floor,
            passthroughOpacity: serverSettings.ar.passthrough_opacity,
            passthroughBrightness: serverSettings.ar.passthrough_brightness,
            passthroughContrast: serverSettings.ar.passthrough_contrast,
            enablePassthroughPortal: serverSettings.ar.enable_passthrough_portal,
            portalSize: serverSettings.ar.portal_size,
            portalEdgeColor: serverSettings.ar.portal_edge_color,
            portalEdgeWidth: serverSettings.ar.portal_edge_width
        };
    }

// ... [Rest of the code remains the same] ...
