// Manages visualization settings received from the server
export class VisualizationSettings {
    constructor() {
        // Default values optimized for basic rendering
        this.settings = {
            // Node colors with increased visibility
            nodeColor: '#FFA500',             // Base orange
            nodeColorNew: '#FFD700',          // Bright gold for very recent files
            nodeColorRecent: '#FFA500',       // Orange for recent files
            nodeColorMedium: '#DAA520',       // Goldenrod for medium-age files
            nodeColorOld: '#CD853F',          // Peru/bronze for old files
            nodeColorCore: '#FFB90F',         // Dark golden for core nodes
            nodeColorSecondary: '#FFC125',    // Golden yellow for secondary nodes
            nodeColorDefault: '#FFD700',      // Gold for default nodes
            
            // Edge settings optimized for visibility
            edgeColor: '#FFD700',             // Golden
            edgeOpacity: 0.8,                 // Increased for better visibility
            edgeWeightNormalization: 12.0,
            edgeMinWidth: 2.0,                // Increased for better visibility
            edgeMaxWidth: 6.0,                // Increased for better visibility
            
            // Node sizes and dimensions (in meters)
            minNodeSize: 0.5,                 // Increased for better visibility
            maxNodeSize: 1.0,                 // Increased for better visibility
            nodeAgeMaxDays: 30,
            
            // Material settings optimized for basic rendering
            material: {
                metalness: 0.1,               // Reduced for less reflection
                roughness: 0.8,               // Increased for more diffuse look
                opacity: 1.0,                 // Full opacity for better visibility
                emissiveMinIntensity: 0.0,    // No emission in basic rendering
                emissiveMaxIntensity: 0.0     // No emission in basic rendering
            },
            
            // Force-directed layout settings
            iterations: 300,
            spring_strength: 0.015,
            repulsion_strength: 1200.0,
            attraction_strength: 0.012,
            damping: 0.85,
            
            // Environment settings
            fogDensity: 0.0002,              // Reduced for better visibility
            
            // Label settings optimized for readability
            labelFontSize: 36,                // Increased for better readability
            labelFontFamily: 'Arial',
            labelPadding: 20,                 // Increased padding
            labelVerticalOffset: 1.2,         // Adjusted for better positioning
            labelCloseOffset: 0.3,
            labelBackgroundColor: 'rgba(0, 0, 0, 0.85)', // More opaque for better contrast
            labelTextColor: 'white',
            labelInfoTextColor: '#ffffff',    // Brighter for better visibility
            labelXRFontSize: 28,
            
            // Geometry settings
            geometryMinSegments: 16,          // Reduced for better performance
            geometryMaxSegments: 32,          // Reduced for better performance
            geometrySegmentPerHyperlink: 0.4, // Reduced for better performance
            
            // Interaction settings
            clickEmissiveBoost: 0.0,          // Disabled for basic rendering
            clickFeedbackDuration: 250
        };

        // Bind the WebSocket message handler
        this.handleServerSettings = this.handleServerSettings.bind(this);
        window.addEventListener('serverSettings', this.handleServerSettings);

        console.log('Visualization settings initialized with optimized defaults');
    }

    handleServerSettings(event) {
        console.log('Received server settings:', event.detail);
        const serverSettings = event.detail;
        
        // Deep merge settings with server values
        this.settings = this.deepMerge(this.settings, {
            ...serverSettings.visualization,
            material: serverSettings.visualization?.material
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
            maxWidth: this.settings.edgeMaxWidth
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
            fogDensity: this.settings.fogDensity
        };
    }
}

// Create and export singleton instance
export const visualizationSettings = new VisualizationSettings();
