// Manages visualization settings received from the server
export class VisualizationSettings {
    constructor() {
        // Initialize empty settings structure - will be populated from server
        this.settings = null;

        // Bind the WebSocket message handler
        this.handleServerSettings = this.handleServerSettings.bind(this);
        window.addEventListener('serverSettings', this.handleServerSettings);

        console.log('Visualization settings initialized - waiting for server settings');
    }

    handleServerSettings(event) {
        console.log('Received server settings:', event.detail);
        
        // Store settings received from server
        this.settings = event.detail;

        console.log('Updated settings with server values');

        // Dispatch event to notify components of updated settings
        window.dispatchEvent(new CustomEvent('visualizationSettingsUpdated', {
            detail: this.settings
        }));
    }

    getSettings() {
        if (!this.settings) {
            console.warn('Settings not yet received from server');
            return null;
        }
        return this.settings;
    }

    getNodeSettings() {
        if (!this.settings?.visualization) {
            console.warn('Visualization settings not yet received from server');
            return null;
        }

        const vis = this.settings.visualization;
        return {
            color: vis.node_color,
            colorNew: vis.node_color_new,
            colorRecent: vis.node_color_recent,
            colorMedium: vis.node_color_medium,
            colorOld: vis.node_color_old,
            colorCore: vis.node_color_core,
            colorSecondary: vis.node_color_secondary,
            colorDefault: vis.node_color_default,
            minNodeSize: vis.min_node_size,
            maxNodeSize: vis.max_node_size,
            material: {
                metalness: vis.node_material_metalness,
                roughness: vis.node_material_roughness,
                clearcoat: vis.node_material_clearcoat,
                clearcoatRoughness: vis.node_material_clearcoat_roughness,
                opacity: vis.node_material_opacity,
                emissiveMinIntensity: vis.node_emissive_min_intensity,
                emissiveMaxIntensity: vis.node_emissive_max_intensity
            },
            ageMaxDays: vis.node_age_max_days,
            geometryMinSegments: vis.geometry_min_segments,
            geometryMaxSegments: vis.geometry_max_segments,
            geometrySegmentPerHyperlink: vis.geometry_segment_per_hyperlink,
            clickEmissiveBoost: vis.click_emissive_boost,
            clickFeedbackDuration: vis.click_feedback_duration
        };
    }

    getEdgeSettings() {
        if (!this.settings?.visualization) {
            console.warn('Visualization settings not yet received from server');
            return null;
        }

        const vis = this.settings.visualization;
        return {
            color: vis.edge_color,
            opacity: vis.edge_opacity,
            weightNormalization: vis.edge_weight_normalization,
            minWidth: vis.edge_min_width,
            maxWidth: vis.edge_max_width
        };
    }

    getLabelSettings() {
        if (!this.settings?.visualization) {
            console.warn('Visualization settings not yet received from server');
            return null;
        }

        const vis = this.settings.visualization;
        return {
            fontSize: vis.label_font_size,
            fontFamily: vis.label_font_family,
            padding: vis.label_padding,
            verticalOffset: vis.label_vertical_offset,
            closeOffset: vis.label_close_offset,
            backgroundColor: vis.label_background_color,
            textColor: vis.label_text_color,
            infoTextColor: vis.label_info_text_color,
            xrFontSize: vis.label_xr_font_size
        };
    }

    getLayoutSettings() {
        if (!this.settings?.visualization) {
            console.warn('Visualization settings not yet received from server');
            return null;
        }

        const vis = this.settings.visualization;
        return {
            iterations: vis.force_directed_iterations,
            spring_strength: vis.force_directed_spring,
            repulsion_strength: vis.force_directed_repulsion,
            attraction_strength: vis.force_directed_attraction,
            damping: vis.force_directed_damping
        };
    }

    getEnvironmentSettings() {
        if (!this.settings?.visualization) {
            console.warn('Visualization settings not yet received from server');
            return null;
        }

        const vis = this.settings.visualization;
        return {
            fogDensity: vis.fog_density
        };
    }

    getBloomSettings() {
        if (!this.settings?.bloom) {
            console.warn('Bloom settings not yet received from server');
            return null;
        }
        return this.settings.bloom;
    }

    getFisheyeSettings() {
        if (!this.settings?.fisheye) {
            console.warn('Fisheye settings not yet received from server');
            return null;
        }
        return this.settings.fisheye;
    }

    updateSettings(settings) {
        // Send settings update to server
        window.dispatchEvent(new CustomEvent('updateSettings', {
            detail: settings
        }));
    }
}

// Create and export singleton instance
export const visualizationSettings = new VisualizationSettings();
