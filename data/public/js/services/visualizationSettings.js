// Manages visualization settings received from the server
export class VisualizationSettings {
    constructor() {
        // Default values matching settings.toml
        this.settings = {
            // Node colors
            nodeColor: '#FFA500',             // Base orange
            nodeColorNew: '#FFD700',          // Bright gold for very recent files
            nodeColorRecent: '#FFA500',       // Orange for recent files
            nodeColorMedium: '#DAA520',       // Goldenrod for medium-age files
            nodeColorOld: '#CD853F',          // Peru/bronze for old files
            nodeColorCore: '#FFB90F',         // Dark golden for core nodes
            nodeColorSecondary: '#FFC125',    // Golden yellow for secondary nodes
            nodeColorDefault: '#FFD700',      // Gold for default nodes
            
            // Edge settings
            edgeColor: '#FFD700',             // Golden
            edgeOpacity: 0.3,                 // Reduced from 0.4
            edgeWeightNormalization: 12.0,
            edgeMinWidth: 1.0,                // Reduced from 1.5
            edgeMaxWidth: 4.0,                // Reduced from 6.0
            
            // Node sizes and dimensions (in meters)
            minNodeSize: 0.3,                 // Increased from 0.15
            maxNodeSize: 0.6,                 // Increased from 0.4
            nodeAgeMaxDays: 30,
            
            // Hologram settings
            hologramColor: '#FFC125',         // Deep golden yellow
            hologramScale: 6.0,
            hologramOpacity: 0.15,
            
            // Material settings
            material: {
                metalness: 0.2,               // Reduced from 0.3
                roughness: 0.7,               // Increased from 0.5
                clearcoat: 0.5,               // Reduced from 0.8
                clearcoatRoughness: 0.2,      // Increased from 0.1
                opacity: 0.9,                 // Reduced from 0.95
                emissiveMinIntensity: 0.1,    // Increased from 0.0
                emissiveMaxIntensity: 0.4     // Increased from 0.3
            },
            
            // Force-directed layout settings
            iterations: 300,
            spring_strength: 0.015,
            repulsion_strength: 1200.0,
            attraction_strength: 0.012,
            damping: 0.85,
            
            // Environment settings
            fogDensity: 0.0005,              // Reduced from 0.001
            
            // Label settings
            labelFontSize: 32,                // Reduced from 42
            labelFontFamily: 'Arial',
            labelPadding: 16,                 // Reduced from 24
            labelVerticalOffset: 0.8,         // Reduced from 2.5
            labelCloseOffset: 0.2,
            labelBackgroundColor: 'rgba(0, 0, 0, 0.7)', // More transparent
            labelTextColor: 'white',
            labelInfoTextColor: '#cccccc',    // Slightly darker for better contrast
            labelXRFontSize: 24,
            
            // Geometry settings
            geometryMinSegments: 24,
            geometryMaxSegments: 48,
            geometrySegmentPerHyperlink: 0.6,
            
            // Interaction settings
            clickEmissiveBoost: 2.5,
            clickFeedbackDuration: 250,
            
            // Bloom settings
            nodeBloomStrength: 0.6,           // Reduced from 0.8
            nodeBloomRadius: 0.4,             // Increased from 0.3
            nodeBloomThreshold: 0.3,          // Increased from 0.2
            edgeBloomStrength: 0.4,           // Reduced from 0.6
            edgeBloomRadius: 0.3,             // Reduced from 0.4
            edgeBloomThreshold: 0.2,          // Increased from 0.1
            environmentBloomStrength: 0.5,     // Reduced from 0.7
            environmentBloomRadius: 0.3,
            environmentBloomThreshold: 0.2,    // Increased from 0.1

            // Fisheye settings
            fisheye: {
                enabled: false,
                strength: 0.5,
                radius: 100.0,
                focusX: 0.0,
                focusY: 0.0,
                focusZ: 0.0
            }
        };

        // Bind the WebSocket message handler
        this.handleServerSettings = this.handleServerSettings.bind(this);
        window.addEventListener('serverSettings', this.handleServerSettings);

        console.log('Visualization settings initialized with defaults from settings.toml');
    }

    handleServerSettings(event) {
        console.log('Received server settings:', event.detail);
        const serverSettings = event.detail;
        
        // Deep merge settings with server values
        this.settings = this.deepMerge(this.settings, {
            ...serverSettings.visualization,
            material: serverSettings.visualization?.material,
            fisheye: serverSettings.fisheye,
            ...serverSettings.bloom
        });

        console.log('Updated settings with server values');

        // Dispatch event to notify components of updated settings
        window.dispatchEvent(new CustomEvent('visualizationSettingsUpdated', {
            detail: this.settings
        }));
    }

    // Deep merge helper function
    deepMerge(target, source) {
        const result = { ...target };
        
        if (source) {
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
        }
        
        return result;
    }

    getSettings() {
        return this.settings;
    }

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

    getBloomSettings() {
        return {
            nodeBloomStrength: this.settings.nodeBloomStrength,
            nodeBloomRadius: this.settings.nodeBloomRadius,
            nodeBloomThreshold: this.settings.nodeBloomThreshold,
            edgeBloomStrength: this.settings.edgeBloomStrength,
            edgeBloomRadius: this.settings.edgeBloomRadius,
            edgeBloomThreshold: this.settings.edgeBloomThreshold,
            environmentBloomStrength: this.settings.environmentBloomStrength,
            environmentBloomRadius: this.settings.environmentBloomRadius,
            environmentBloomThreshold: this.settings.environmentBloomThreshold
        };
    }
}

// Create and export singleton instance
export const visualizationSettings = new VisualizationSettings();
