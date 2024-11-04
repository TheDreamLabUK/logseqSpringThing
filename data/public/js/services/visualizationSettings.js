// Manages visualization settings received from the server and provides defaults
export class VisualizationSettings {
    constructor() {
        // Default values matching .env_template
        this.settings = {
            // Node settings
            nodeColor: process.env.NODE_COLOR || '0x1A0B31',
            nodeSizeScalingFactor: parseInt(process.env.NODE_SIZE_SCALING_FACTOR) || 5,
            
            // Edge settings
            edgeColor: process.env.EDGE_COLOR || '0xff0000',
            edgeOpacity: parseFloat(process.env.EDGE_OPACITY) || 0.3,
            
            // Hologram settings
            hologramColor: process.env.HOLOGRAM_COLOR || '0xFFD700',
            hologramScale: parseInt(process.env.HOLOGRAM_SCALE) || 5,
            hologramOpacity: parseFloat(process.env.HOLOGRAM_OPACITY) || 0.1,
            
            // Label settings
            labelFontSize: parseInt(process.env.LABEL_FONT_SIZE) || 36,
            
            // Environment settings
            fogDensity: parseFloat(process.env.FOG_DENSITY) || 0.002,
            
            // Force-directed layout settings
            forceDirectedIterations: parseInt(process.env.FORCE_DIRECTED_ITERATIONS) || 100,
            forceDirectedSpring: parseFloat(process.env.FORCE_DIRECTED_SPRING) || 0.1,
            forceDirectedRepulsion: parseFloat(process.env.FORCE_DIRECTED_REPULSION) || 1000.0,
            forceDirectedAttraction: parseFloat(process.env.FORCE_DIRECTED_ATTRACTION) || 0.01,
            forceDirectedDamping: parseFloat(process.env.FORCE_DIRECTED_DAMPING) || 0.8,
            
            // Bloom settings
            nodeBloomStrength: parseFloat(process.env.NODE_BLOOM_STRENGTH) || 0.1,
            nodeBloomRadius: parseFloat(process.env.NODE_BLOOM_RADIUS) || 0.1,
            nodeBloomThreshold: parseFloat(process.env.NODE_BLOOM_THRESHOLD) || 0.0,
            edgeBloomStrength: parseFloat(process.env.EDGE_BLOOM_STRENGTH) || 0.2,
            edgeBloomRadius: parseFloat(process.env.EDGE_BLOOM_RADIUS) || 0.3,
            edgeBloomThreshold: parseFloat(process.env.EDGE_BLOOM_THRESHOLD) || 0.0,
            environmentBloomStrength: parseFloat(process.env.ENVIRONMENT_BLOOM_STRENGTH) || 0.5,
            environmentBloomRadius: parseFloat(process.env.ENVIRONMENT_BLOOM_RADIUS) || 0.1,
            environmentBloomThreshold: parseFloat(process.env.ENVIRONMENT_BLOOM_THRESHOLD) || 0.0
        };

        // Bind the WebSocket message handler
        this.handleServerSettings = this.handleServerSettings.bind(this);
        window.addEventListener('serverSettings', this.handleServerSettings);
    }

    handleServerSettings(event) {
        const serverSettings = event.detail;
        // Update settings with values from server, maintaining defaults for any missing values
        this.settings = {
            ...this.settings,
            ...serverSettings.visualization,
            ...serverSettings.bloom
        };

        // Dispatch event to notify components of updated settings
        window.dispatchEvent(new CustomEvent('visualizationSettingsUpdated', {
            detail: this.settings
        }));
    }

    getSettings() {
        return this.settings;
    }

    // Get settings for specific components
    getNodeSettings() {
        return {
            color: this.settings.nodeColor,
            sizeScalingFactor: this.settings.nodeSizeScalingFactor,
            bloomStrength: this.settings.nodeBloomStrength,
            bloomRadius: this.settings.nodeBloomRadius,
            bloomThreshold: this.settings.nodeBloomThreshold
        };
    }

    getEdgeSettings() {
        return {
            color: this.settings.edgeColor,
            opacity: this.settings.edgeOpacity,
            bloomStrength: this.settings.edgeBloomStrength,
            bloomRadius: this.settings.edgeBloomRadius,
            bloomThreshold: this.settings.edgeBloomThreshold
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
            iterations: this.settings.forceDirectedIterations,
            spring: this.settings.forceDirectedSpring,
            repulsion: this.settings.forceDirectedRepulsion,
            attraction: this.settings.forceDirectedAttraction,
            damping: this.settings.forceDirectedDamping
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
}

// Create and export a singleton instance
export const visualizationSettings = new VisualizationSettings();
