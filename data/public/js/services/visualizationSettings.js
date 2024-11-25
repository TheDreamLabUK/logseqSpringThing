// Manages visualization settings received from the server
export class VisualizationSettings {
    constructor() {
        // Default values optimized for both desktop and XR rendering
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
            edgeMinWidth: 0.002,              // Adjusted for XR scale
            edgeMaxWidth: 0.006,              // Adjusted for XR scale
            
            // Node sizes and dimensions (in meters)
            minNodeSize: 0.05,                // Adjusted for XR scale
            maxNodeSize: 0.1,                 // Adjusted for XR scale
            nodeAgeMaxDays: 30,
            
            // Material settings optimized for XR
            material: {
                metalness: 0.2,               // Slightly increased for better depth perception in XR
                roughness: 0.7,               // Adjusted for better visual quality in XR
                opacity: 1.0,
                emissiveMinIntensity: 0.2,    // Added subtle emission for better visibility in XR
                emissiveMaxIntensity: 0.5     // Maximum emission for highlighted nodes
            },
            
            // Force-directed layout settings
            iterations: 300,
            spring_strength: 0.015,
            repulsion_strength: 1200.0,
            attraction_strength: 0.012,
            damping: 0.85,
            
            // Environment settings
            fogDensity: 0.0001,              // Reduced for better depth perception in XR
            
            // Label settings optimized for both desktop and XR
            labelFontSize: 36,                // Desktop font size
            labelFontFamily: 'Arial',
            labelPadding: 20,
            labelVerticalOffset: 1.2,
            labelCloseOffset: 0.3,
            labelBackgroundColor: 'rgba(0, 0, 0, 0.85)',
            labelTextColor: 'white',
            labelInfoTextColor: '#ffffff',
            labelXRFontSize: 24,              // Smaller font size for XR
            
            // XR-specific settings
            xr: {
                nodeScale: 0.1,               // Scale factor for nodes in XR
                labelScale: 0.5,              // Scale factor for labels in XR
                interactionRadius: 0.2,       // Radius for XR controller interaction
                hapticStrength: 0.5,          // Strength of haptic feedback
                hapticDuration: 50,           // Duration of haptic feedback in ms
                minInteractionDistance: 0.1,  // Minimum distance for interaction
                maxInteractionDistance: 5.0   // Maximum distance for interaction
            },
            
            // Geometry settings optimized for performance
            geometryMinSegments: 8,           // Reduced for better performance
            geometryMaxSegments: 16,          // Reduced for better performance
            geometrySegmentPerHyperlink: 0.2, // Reduced for better performance
            
            // Interaction settings
            clickEmissiveBoost: 0.5,          // Added for visual feedback
            clickFeedbackDuration: 250
        };

        // Bind the WebSocket message handler
        this.handleServerSettings = this.handleServerSettings.bind(this);
        window.addEventListener('serverSettings', this.handleServerSettings);

        console.log('Visualization settings initialized with XR-optimized defaults');
    }

    handleServerSettings(event) {
        console.log('Received server settings:', event.detail);
        const serverSettings = event.detail;
        
        // Deep merge settings with server values
        this.settings = this.deepMerge(this.settings, {
            ...serverSettings.visualization,
            material: serverSettings.visualization?.material,
            xr: serverSettings.visualization?.xr // Ensure XR settings are merged
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

    getXRSettings() {
        return this.settings.xr || {
            nodeScale: 0.1,
            labelScale: 0.5,
            interactionRadius: 0.2,
            hapticStrength: 0.5,
            hapticDuration: 50,
            minInteractionDistance: 0.1,
            maxInteractionDistance: 5.0
        };
    }
}

// Create and export singleton instance
export const visualizationSettings = new VisualizationSettings();
