<!-- Previous template and style sections remain the same -->

<script>
import { defineComponent, ref, reactive, onMounted } from 'vue';

export default defineComponent({
    name: 'ControlPanel',
    props: {
        websocketService: {
            type: Object,
            required: true
        }
    },
    setup(props, { emit }) {
        // Panel state
        const isHidden = ref(false);
        const audioInitialized = ref(false);
        const simulationMode = ref('remote');

        // Collapsed state for groups
        const collapsedGroups = reactive({
            audio: true,
            nodeAppearance: true,
            edgeAppearance: true,
            bloom: true,
            physics: true,
            environment: true,
            fisheye: true
        });

        // Node appearance controls
        const nodeColors = reactive([
            { name: 'nodeColor', label: 'Base Node Color', value: '#1A0B31' },
            { name: 'nodeColorNew', label: 'New Nodes', value: '#00ff88' },
            { name: 'nodeColorRecent', label: 'Recent Nodes', value: '#4444ff' },
            { name: 'nodeColorMedium', label: 'Medium Age', value: '#ffaa00' },
            { name: 'nodeColorOld', label: 'Old Nodes', value: '#ff4444' },
            { name: 'nodeColorCore', label: 'Core Nodes', value: '#ffa500' },
            { name: 'nodeColorSecondary', label: 'Secondary Nodes', value: '#00ffff' }
        ]);

        const materialProperties = reactive([
            { name: 'nodeMaterialMetalness', label: 'Metalness', value: 0.2, min: 0, max: 1, step: 0.1 },
            { name: 'nodeMaterialRoughness', label: 'Roughness', value: 0.2, min: 0, max: 1, step: 0.1 },
            { name: 'nodeMaterialClearcoat', label: 'Clearcoat', value: 0.3, min: 0, max: 1, step: 0.1 },
            { name: 'nodeMaterialClearcoatRoughness', label: 'Clearcoat Roughness', value: 0.2, min: 0, max: 1, step: 0.1 },
            { name: 'nodeMaterialOpacity', label: 'Opacity', value: 0.9, min: 0, max: 1, step: 0.1 },
            { name: 'nodeEmissiveMin', label: 'Min Emissive', value: 0.3, min: 0, max: 1, step: 0.1 },
            { name: 'nodeEmissiveMax', label: 'Max Emissive', value: 1.0, min: 0, max: 2, step: 0.1 }
        ]);

        const sizeControls = reactive([
            { name: 'minNodeSize', label: 'Minimum Size', value: 0.1, min: 0.05, max: 0.5, step: 0.05 },
            { name: 'maxNodeSize', label: 'Maximum Size', value: 0.3, min: 0.1, max: 1.0, step: 0.1 }
        ]);

        // Edge appearance controls
        const edgeControls = reactive([
            { name: 'edgeColor', label: 'Edge Color', value: '#ff0000', type: 'color' },
            { name: 'edgeOpacity', label: 'Opacity', value: 0.3, min: 0, max: 1, step: 0.1 },
            { name: 'edgeWeightNorm', label: 'Weight Normalization', value: 10.0, min: 1, max: 20, step: 0.5 },
            { name: 'edgeMinWidth', label: 'Minimum Width', value: 1.0, min: 0.5, max: 5, step: 0.5 },
            { name: 'edgeMaxWidth', label: 'Maximum Width', value: 5.0, min: 1, max: 10, step: 0.5 }
        ]);

        // Bloom effect controls
        const bloomControls = reactive({
            nodes: {
                label: 'Node Bloom',
                controls: [
                    { name: 'nodeBloomStrength', label: 'Strength', value: 0.1, min: 0, max: 2, step: 0.1 },
                    { name: 'nodeBloomRadius', label: 'Radius', value: 0.1, min: 0, max: 1, step: 0.1 },
                    { name: 'nodeBloomThreshold', label: 'Threshold', value: 0.0, min: 0, max: 1, step: 0.1 }
                ]
            },
            edges: {
                label: 'Edge Bloom',
                controls: [
                    { name: 'edgeBloomStrength', label: 'Strength', value: 0.2, min: 0, max: 2, step: 0.1 },
                    { name: 'edgeBloomRadius', label: 'Radius', value: 0.3, min: 0, max: 1, step: 0.1 },
                    { name: 'edgeBloomThreshold', label: 'Threshold', value: 0.0, min: 0, max: 1, step: 0.1 }
                ]
            },
            environment: {
                label: 'Environment Bloom',
                controls: [
                    { name: 'envBloomStrength', label: 'Strength', value: 0.5, min: 0, max: 2, step: 0.1 },
                    { name: 'envBloomRadius', label: 'Radius', value: 0.1, min: 0, max: 1, step: 0.1 },
                    { name: 'envBloomThreshold', label: 'Threshold', value: 0.0, min: 0, max: 1, step: 0.1 }
                ]
            }
        });

        // Physics simulation controls
        const physicsControls = reactive([
            { name: 'iterations', label: 'Iterations', value: 250, min: 100, max: 500, step: 10 },
            { name: 'springStrength', label: 'Spring Strength', value: 0.01, min: 0.001, max: 0.1, step: 0.001 },
            { name: 'repulsionStrength', label: 'Repulsion', value: 1000.0, min: 100, max: 2000, step: 100 },
            { name: 'attractionStrength', label: 'Attraction', value: 0.01, min: 0.001, max: 0.1, step: 0.001 },
            { name: 'damping', label: 'Damping', value: 0.8, min: 0.1, max: 1.0, step: 0.1 }
        ]);

        // Environment controls
        const hologramControls = reactive([
            { name: 'hologramColor', label: 'Color', value: '#FFD700', type: 'color' },
            { name: 'hologramScale', label: 'Scale', value: 5, min: 1, max: 10, step: 1 },
            { name: 'hologramOpacity', label: 'Opacity', value: 0.1, min: 0, max: 1, step: 0.05 }
        ]);

        const fogDensity = ref(0.002);

        // Fisheye controls
        const fisheyeEnabled = ref(false);
        const fisheyeControls = reactive([
            { name: 'fisheyeStrength', label: 'Strength', value: 0.5, min: 0, max: 1, step: 0.1 },
            { name: 'fisheyeRadius', label: 'Radius', value: 100.0, min: 10, max: 200, step: 10 },
            { name: 'fisheyeFocusX', label: 'Focus X', value: 0.0, min: -100, max: 100, step: 1 },
            { name: 'fisheyeFocusY', label: 'Focus Y', value: 0.0, min: -100, max: 100, step: 1 },
            { name: 'fisheyeFocusZ', label: 'Focus Z', value: 0.0, min: -100, max: 100, step: 1 }
        ]);

        // Methods
        const togglePanel = () => {
            isHidden.value = !isHidden.value;
        };

        const toggleGroup = (group) => {
            collapsedGroups[group] = !collapsedGroups[group];
        };

        const initializeAudio = async () => {
            if (props.websocketService) {
                try {
                    await props.websocketService.initAudio();
                    audioInitialized.value = true;
                    console.log('Audio system initialized successfully');
                } catch (error) {
                    console.error('Failed to initialize audio:', error);
                    audioInitialized.value = false;
                }
            }
        };

        const setSimulationMode = () => {
            if (props.websocketService) {
                props.websocketService.setSimulationMode(simulationMode.value);
            }
        };

        const emitChange = (name, value) => {
            emit('control-change', { name, value });
            if (props.websocketService) {
                props.websocketService.send({
                    type: 'settingUpdate',
                    setting: name,
                    value: value
                });
            }
        };

        const saveSettings = () => {
            if (props.websocketService) {
                const settings = {
                    visualization: {
                        node_color: nodeColors.find(c => c.name === 'nodeColor')?.value,
                        node_color_new: nodeColors.find(c => c.name === 'nodeColorNew')?.value,
                        node_color_recent: nodeColors.find(c => c.name === 'nodeColorRecent')?.value,
                        node_color_medium: nodeColors.find(c => c.name === 'nodeColorMedium')?.value,
                        node_color_old: nodeColors.find(c => c.name === 'nodeColorOld')?.value,
                        node_color_core: nodeColors.find(c => c.name === 'nodeColorCore')?.value,
                        node_color_secondary: nodeColors.find(c => c.name === 'nodeColorSecondary')?.value,
                        edge_color: edgeControls.find(c => c.name === 'edgeColor')?.value,
                        edge_opacity: edgeControls.find(c => c.name === 'edgeOpacity')?.value || 0.4,
                        edge_weight_normalization: edgeControls.find(c => c.name === 'edgeWeightNorm')?.value || 12.0,
                        edge_min_width: edgeControls.find(c => c.name === 'edgeMinWidth')?.value || 1.5,
                        edge_max_width: edgeControls.find(c => c.name === 'edgeMaxWidth')?.value || 6.0,
                        node_material_metalness: materialProperties.find(p => p.name === 'nodeMaterialMetalness')?.value || 0.3,
                        node_material_roughness: materialProperties.find(p => p.name === 'nodeMaterialRoughness')?.value || 0.5,
                        node_material_clearcoat: materialProperties.find(p => p.name === 'nodeMaterialClearcoat')?.value || 0.8,
                        node_material_clearcoat_roughness: materialProperties.find(p => p.name === 'nodeMaterialClearcoatRoughness')?.value || 0.1,
                        node_material_opacity: materialProperties.find(p => p.name === 'nodeMaterialOpacity')?.value || 0.95,
                        node_emissive_min_intensity: materialProperties.find(p => p.name === 'nodeEmissiveMin')?.value || 0.0,
                        node_emissive_max_intensity: materialProperties.find(p => p.name === 'nodeEmissiveMax')?.value || 0.3,
                        min_node_size: sizeControls.find(s => s.name === 'minNodeSize')?.value || 0.15,
                        max_node_size: sizeControls.find(s => s.name === 'maxNodeSize')?.value || 0.4,
                        force_directed_iterations: physicsControls.find(p => p.name === 'iterations')?.value || 300,
                        force_directed_spring: physicsControls.find(p => p.name === 'springStrength')?.value || 0.015,
                        force_directed_repulsion: physicsControls.find(p => p.name === 'repulsionStrength')?.value || 1200.0,
                        force_directed_attraction: physicsControls.find(p => p.name === 'attractionStrength')?.value || 0.012,
                        force_directed_damping: physicsControls.find(p => p.name === 'damping')?.value || 0.85,
                        hologram_color: hologramControls.find(h => h.name === 'hologramColor')?.value || '#FFC125',
                        hologram_scale: hologramControls.find(h => h.name === 'hologramScale')?.value || 6.0,
                        hologram_opacity: hologramControls.find(h => h.name === 'hologramOpacity')?.value || 0.15,
                        fog_density: fogDensity.value || 0.001
                    },
                    bloom: {
                        node_bloom_strength: bloomControls.nodes.controls.find(c => c.name === 'nodeBloomStrength')?.value || 0.8,
                        node_bloom_radius: bloomControls.nodes.controls.find(c => c.name === 'nodeBloomRadius')?.value || 0.3,
                        node_bloom_threshold: bloomControls.nodes.controls.find(c => c.name === 'nodeBloomThreshold')?.value || 0.2,
                        edge_bloom_strength: bloomControls.edges.controls.find(c => c.name === 'edgeBloomStrength')?.value || 0.6,
                        edge_bloom_radius: bloomControls.edges.controls.find(c => c.name === 'edgeBloomRadius')?.value || 0.4,
                        edge_bloom_threshold: bloomControls.edges.controls.find(c => c.name === 'edgeBloomThreshold')?.value || 0.1,
                        environment_bloom_strength: bloomControls.environment.controls.find(c => c.name === 'envBloomStrength')?.value || 0.7,
                        environment_bloom_radius: bloomControls.environment.controls.find(c => c.name === 'envBloomRadius')?.value || 0.3,
                        environment_bloom_threshold: bloomControls.environment.controls.find(c => c.name === 'envBloomThreshold')?.value || 0.1
                    },
                    fisheye: {
                        enabled: fisheyeEnabled.value,
                        strength: fisheyeControls.find(c => c.name === 'fisheyeStrength')?.value || 0.5,
                        radius: fisheyeControls.find(c => c.name === 'fisheyeRadius')?.value || 100.0,
                        focus_x: fisheyeControls.find(c => c.name === 'fisheyeFocusX')?.value || 0.0,
                        focus_y: fisheyeControls.find(c => c.name === 'fisheyeFocusY')?.value || 0.0,
                        focus_z: fisheyeControls.find(c => c.name === 'fisheyeFocusZ')?.value || 0.0
                    }
                };

                console.log('Saving settings to TOML:', settings);
                
                props.websocketService.send({
                    type: 'saveSettings',
                    settings
                });
            }
        };

        // Initialize settings from server
        onMounted(() => {
            if (props.websocketService) {
                props.websocketService.on('serverSettings', (settings) => {
                    console.log('Received server settings:', settings);

                    // Helper function to update reactive values
                    const updateValue = (obj, path, value) => {
                        if (!value) return;
                        const target = path.split('.').reduce((acc, key) => acc?.[key], obj);
                        if (target && 'value' in target) {
                            target.value = value;
                        }
                    };

                    // Update visualization settings
                    if (settings.visualization) {
                        // Update node colors - no need for hex conversion
                        nodeColors.forEach(color => {
                            const settingName = color.name
                                .replace(/([A-Z])/g, '_$1')
                                .toLowerCase();
                            updateValue(nodeColors, color.name, settings.visualization[settingName]);
                        });

                        // Update material properties
                        materialProperties.forEach(prop => {
                            const settingName = prop.name
                                .replace(/([A-Z])/g, '_$1')
                                .toLowerCase();
                            updateValue(materialProperties, prop.name, settings.visualization[settingName]);
                        });

                        // Update other controls
                        updateValue(sizeControls, 'minNodeSize', settings.visualization.min_node_size);
                        updateValue(sizeControls, 'maxNodeSize', settings.visualization.max_node_size);
                        updateValue(edgeControls, 'edgeColor', settings.visualization.edge_color);
                        updateValue(hologramControls, 'hologramColor', settings.visualization.hologram_color);
                        fogDensity.value = settings.visualization.fog_density ?? fogDensity.value;
                    }

                    // Update bloom settings
                    if (settings.bloom) {
                        Object.entries(bloomControls).forEach(([group, config]) => {
                            config.controls.forEach(control => {
                                const settingName = control.name
                                    .replace(/([A-Z])/g, '_$1')
                                    .toLowerCase();
                                updateValue(bloomControls[group].controls, control.name, settings.bloom[settingName]);
                            });
                        });
                    }

                    // Update fisheye settings
                    if (settings.fisheye) {
                        fisheyeEnabled.value = settings.fisheye.enabled ?? fisheyeEnabled.value;
                        fisheyeControls.forEach(control => {
                            const settingName = control.name
                                .replace(/([A-Z])/g, '_$1')
                                .toLowerCase();
                            updateValue(fisheyeControls, control.name, settings.fisheye[settingName]);
                        });
                    }
                });
            }
        });

        return {
            isHidden,
            audioInitialized,
            simulationMode,
            collapsedGroups,
            nodeColors,
            nodeMaterialMetalness,
            minNodeSize,
            edgeControls,
            bloomControls,
            physicsControls,
            hologramControls,
            fogDensity,
            fisheyeEnabled,
            fisheyeControls,
            togglePanel,
            toggleGroup,
            initializeAudio,
            setSimulationMode,
            emitChange,
            saveSettings
        };
    }
});
</script>

<style scoped>
#control-panel {
    position: fixed;
    top: 20px;
    right: 0;
    width: 300px;
    max-height: 90vh;
    background-color: rgba(20, 20, 20, 0.9);
    color: #ffffff;
    border-radius: 10px 0 0 10px;
    overflow-y: auto;
    z-index: 1000;
    transition: transform 0.3s ease-in-out;
    box-shadow: -2px 0 10px rgba(0, 0, 0, 0.5);
}

#control-panel.hidden {
    transform: translateX(calc(100% - 40px));
}

.toggle-button {
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%) rotate(-90deg);
    transform-origin: left center;
    background-color: rgba(20, 20, 20, 0.9);
    color: #ffffff;
    border: none;
    padding: 8px 16px;
    cursor: pointer;
    border-radius: 5px 5px 0 0;
    font-size: 0.9em;
    white-space: nowrap;
}

.panel-content {
    padding: 20px;
}

.control-group {
    margin-bottom: 16px;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    overflow: hidden;
}

.group-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    background-color: rgba(255, 255, 255, 0.1);
    cursor: pointer;
}

.group-header h3 {
    margin: 0;
    font-size: 1em;
    font-weight: 500;
}

.group-content {
    padding: 12px;
}

.control-item {
    margin-bottom: 12px;
}

.control-item label {
    display: block;
    margin-bottom: 4px;
    font-size: 0.9em;
    color: #cccccc;
}

.control-item input[type="range"] {
    width: 100%;
    height: 6px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    -webkit-appearance: none;
}

.control-item input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background-color: #ffffff;
    border-radius: 50%;
    cursor: pointer;
}

.control-item input[type="color"] {
    width: 100%;
    height: 30px;
    border: none;
    border-radius: 4px;
    background-color: transparent;
}

.range-value {
    float: right;
    font-size: 0.8em;
    color: #999999;
}

.save-button {
    width: 100%;
    padding: 12px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.2s;
}

.save-button:hover {
    background-color: #45a049;
}

select {
    width: 100%;
    padding: 8px;
    background-color: rgba(255, 255, 255, 0.1);
    color: #ffffff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

select option {
    background-color: #1a1a1a;
    color: #ffffff;
}

/* Add new styles for sub-groups */
.sub-group {
    margin: 10px 0;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
}

.sub-group h4 {
    margin: 0 0 10px 0;
    font-size: 0.9em;
    color: #aaa;
}

/* Audio status styles */
.audio-status {
    text-align: center;
    padding: 10px;
}

.status-indicator {
    padding: 5px 10px;
    border-radius: 3px;
    font-size: 0.9em;
}

.status-indicator.enabled {
    background-color: rgba(40, 167, 69, 0.2);
    color: #28a745;
}

/* Save button styling */
.save-button {
    width: 100%;
    padding: 12px;
    margin-top: 20px;
    background-color: #28a745;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.2s;
}

.save-button:hover {
    background-color: #218838;
}
</style>
