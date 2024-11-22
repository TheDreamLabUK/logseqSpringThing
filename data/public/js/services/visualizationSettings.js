// Manages visualization settings received from the server
export class VisualizationSettings {
    constructor() {
        // Default values matching settings.toml
        this.settings = {
            // Node colors
            nodeColor: process.env.NODE_COLOR || '#1A0B31',
            nodeColorNew: process.env.NODE_COLOR_NEW || '#00ff88',
            nodeColorRecent: process.env.NODE_COLOR_RECENT || '#4444ff',
            nodeColorMedium: process.env.NODE_COLOR_MEDIUM || '#ffaa00',
            nodeColorOld: process.env.NODE_COLOR_OLD || '#ff4444',
            nodeColorCore: process.env.NODE_COLOR_CORE || '#ffa500',
            nodeColorSecondary: process.env.NODE_COLOR_SECONDARY || '#00ffff',
            nodeColorDefault: process.env.NODE_COLOR_DEFAULT || '#00ff00',
            
            // Edge settings
            edgeColor: process.env.EDGE_COLOR || '#ff0000',
            edgeOpacity: parseFloat(process.env.EDGE_OPACITY) || 0.3,
            edgeWeightNormalization: parseFloat(process.env.EDGE_WEIGHT_NORMALIZATION) || 10.0,
            edgeMinWidth: parseFloat(process.env.EDGE_MIN_WIDTH) || 1.0,
            edgeMaxWidth: parseFloat(process.env.EDGE_MAX_WIDTH) || 5.0,
            
            // Node sizes and dimensions (in meters)
            minNodeSize: parseFloat(process.env.MIN_NODE_SIZE) || 0.1,  // 10cm
            maxNodeSize: parseFloat(process.env.MAX_NODE_SIZE) || 0.3,  // 30cm
            nodeAgeMaxDays: parseInt(process.env.NODE_AGE_MAX_DAYS) || 30,
            
            // Hologram settings
            hologramColor: process.env.HOLOGRAM_COLOR || '#FFD700',
            hologramScale: parseInt(process.env.HOLOGRAM_SCALE) || 5,
            hologramOpacity: parseFloat(process.env.HOLOGRAM_OPACITY) || 0.1,
            
            // Label settings
            labelFontSize: parseInt(process.env.LABEL_FONT_SIZE) || 36,
            labelFontFamily: process.env.LABEL_FONT_FAMILY || 'Arial',
            labelPadding: parseInt(process.env.LABEL_PADDING) || 20,
            labelVerticalOffset: parseFloat(process.env.LABEL_VERTICAL_OFFSET) || 2.0,
            labelCloseOffset: parseFloat(process.env.LABEL_CLOSE_OFFSET) || 0.2,
            labelBackgroundColor: process.env.LABEL_BACKGROUND_COLOR || 'rgba(0, 0, 0, 0.8)',
            labelTextColor: process.env.LABEL_TEXT_COLOR || 'white',
            labelInfoTextColor: process.env.LABEL_INFO_TEXT_COLOR || 'lightgray',
            labelXRFontSize: parseInt(process.env.LABEL_XR_FONT_SIZE) || 24,
            
            // Environment settings
            fogDensity: parseFloat(process.env.FOG_DENSITY) || 0.002,
            
            // Geometry settings
            geometryMinSegments: parseInt(process.env.GEOMETRY_MIN_SEGMENTS) || 16,
            geometryMaxSegments: parseInt(process.env.GEOMETRY_MAX_SEGMENTS) || 32,
            geometrySegmentPerHyperlink: parseFloat(process.env.GEOMETRY_SEGMENT_PER_HYPERLINK) || 0.5,
            
            // Interaction settings
            clickEmissiveBoost: parseFloat(process.env.CLICK_EMISSIVE_BOOST) || 2.0,
            clickFeedbackDuration: parseInt(process.env.CLICK_FEEDBACK_DURATION) || 200,
            
            // Material settings
            material: {
                metalness: parseFloat(process.env.NODE_MATERIAL_METALNESS) || 0.2,
                roughness: parseFloat(process.env.NODE_MATERIAL_ROUGHNESS) || 0.2,
                clearcoat: parseFloat(process.env.NODE_MATERIAL_CLEARCOAT) || 0.3,
                clearcoatRoughness: parseFloat(process.env.NODE_MATERIAL_CLEARCOAT_ROUGHNESS) || 0.2,
                opacity: parseFloat(process.env.NODE_MATERIAL_OPACITY) || 0.9,
                emissiveMinIntensity: parseFloat(process.env.NODE_EMISSIVE_MIN_INTENSITY) || 0.3,
                emissiveMaxIntensity: parseFloat(process.env.NODE_EMISSIVE_MAX_INTENSITY) || 1.0
            },
            
            // Force-directed layout settings
            iterations: parseInt(process.env.FORCE_DIRECTED_ITERATIONS) || 250,
            spring_strength: parseFloat(process.env.FORCE_DIRECTED_SPRING) || 0.01,
            repulsion_strength: parseFloat(process.env.FORCE_DIRECTED_REPULSION) || 1000.0,
            attraction_strength: parseFloat(process.env.FORCE_DIRECTED_ATTRACTION) || 0.01,
            damping: parseFloat(process.env.FORCE_DIRECTED_DAMPING) || 0.8,
            
            // Bloom settings
            nodeBloomStrength: parseFloat(process.env.NODE_BLOOM_STRENGTH) || 0.1,
            nodeBloomRadius: parseFloat(process.env.NODE_BLOOM_RADIUS) || 0.1,
            nodeBloomThreshold: parseFloat(process.env.NODE_BLOOM_THRESHOLD) || 0.0,
            edgeBloomStrength: parseFloat(process.env.EDGE_BLOOM_STRENGTH) || 0.2,
            edgeBloomRadius: parseFloat(process.env.EDGE_BLOOM_RADIUS) || 0.3,
            edgeBloomThreshold: parseFloat(process.env.EDGE_BLOOM_THRESHOLD) || 0.0,
            environmentBloomStrength: parseFloat(process.env.ENVIRONMENT_BLOOM_STRENGTH) || 0.5,
            environmentBloomRadius: parseFloat(process.env.ENVIRONMENT_BLOOM_RADIUS) || 0.1,
            environmentBloomThreshold: parseFloat(process.env.ENVIRONMENT_BLOOM_THRESHOLD) || 0.0,

            // Fisheye settings
            fisheye: {
                enabled: process.env.FISHEYE_ENABLED === 'true',
                strength: parseFloat(process.env.FISHEYE_STRENGTH) || 0.5,
                radius: parseFloat(process.env.FISHEYE_RADIUS) || 100.0,
                focusX: parseFloat(process.env.FISHEYE_FOCUS_X) || 0.0,
                focusY: parseFloat(process.env.FISHEYE_FOCUS_Y) || 0.0,
                focusZ: parseFloat(process.env.FISHEYE_FOCUS_Z) || 0.0
            }
        };

        // Bind the WebSocket message handler
        this.handleServerSettings = this.handleServerSettings.bind(this);
        window.addEventListener('serverSettings', this.handleServerSettings);
    }

    handleServerSettings(event) {
        const serverSettings = event.detail;
        
        // Deep merge settings with server values
        this.settings = this.deepMerge(this.settings, {
            ...serverSettings.visualization,
            material: serverSettings.visualization?.material,
            fisheye: serverSettings.fisheye,
            ...serverSettings.bloom
        });

        // Dispatch event to notify components of updated settings
        window.dispatchEvent(new CustomEvent('visualizationSettingsUpdated', {
            detail: this.settings
        }));
    }

    // Deep merge helper function
    deepMerge(target, source) {
        const result = { ...target };
        
        Object.keys(source).forEach(key => {
            if (source[key] instanceof Object && !Array.isArray(source[key])) {
                if (key in target) {
                    result[key] = this.deepMerge(target[key], source[key]);
                } else {
                    result[key] = { ...source[key] };
                }
            } else if (source[key] !== undefined) {
                result[key] = source[key];
            }
        });
        
        return result;
    }

    getSettings() {
        return this.settings;
    }

    // Get settings for specific components
    getNodeSettings() {
        return {
            color: this.settings.nodeColor,
            colorNew: this.settings.nodeColorNew,
            colorRecent: this.settings.nodeColorRecent,
            colorMedium: this.settings.nodeColorMedium,
            colorOld: this.settings.nodeColorOld,
            colorCore: this.settings.nodeColorCore,
            colorSecondary: this.settings.nodeColorSecondary,
            colorDefault: this.settings.nodeColorDefault,
            minNodeSize: this.settings.minNodeSize,
            maxNodeSize: this.settings.maxNodeSize,
            material: this.settings.material,
            ageMaxDays: this.settings.nodeAgeMaxDays,
            geometryMinSegments: this.settings.geometryMinSegments,
            geometryMaxSegments: this.settings.geometryMaxSegments,
            geometrySegmentPerHyperlink: this.settings.geometrySegmentPerHyperlink,
            clickEmissiveBoost: this.settings.clickEmissiveBoost,
            clickFeedbackDuration: this.settings.clickFeedbackDuration
        };
    }

    getEdgeSettings() {
        return {
            color: this.settings.edgeColor,
            opacity: this.settings.edgeOpacity,
            weightNormalization: this.settings.edgeWeightNormalization,
            minWidth: this.settings.edgeMinWidth,
            maxWidth: this.settings.edgeMaxWidth,
            bloomStrength: this.settings.edgeBloomStrength,
            bloomRadius: this.settings.edgeBloomRadius,
            bloomThreshold: this.settings.edgeBloomThreshold
        };
    }

    getLabelSettings() {
        return {
            fontSize: this.settings.labelFontSize,
            fontFamily: this.settings.labelFontFamily,
            padding: this.settings.labelPadding,
            verticalOffset: this.settings.labelVerticalOffset,
            closeOffset: this.settings.labelCloseOffset,
            backgroundColor: this.settings.labelBackgroundColor,
            textColor: this.settings.labelTextColor,
            infoTextColor: this.settings.labelInfoTextColor,
            xrFontSize: this.settings.labelXRFontSize
        };
    }

    getHologramSettings() {
        return {
            color: this.settings.hologramColor,
            scale: this.settings.hologramScale,
            opacity: this.settings.hologramOpacity
        };
    }

    getLayoutSettings() {
        return {
            iterations: this.settings.iterations,
            spring_strength: this.settings.spring_strength,
            repulsion_strength: this.settings.repulsion_strength,
            attraction_strength: this.settings.attraction_strength,
            damping: this.settings.damping
        };
    }

    getEnvironmentSettings() {
        return {
            fogDensity: this.settings.fogDensity,
            bloomStrength: this.settings.environmentBloomStrength,
            bloomRadius: this.settings.environmentBloomRadius,
            bloomThreshold: this.settings.environmentBloomThreshold
        };
    }

    getFisheyeSettings() {
        return this.settings.fisheye;
    }
}

// Create and export a singleton instance
export const visualizationSettings = new VisualizationSettings();
