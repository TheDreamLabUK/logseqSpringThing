<template>
    <div id="control-panel" :class="{ hidden: isHidden }">
        <button @click="togglePanel" class="toggle-button">
            {{ isHidden ? '>' : '<' }}
        </button>
        <div class="panel-content" v-show="!isHidden">
            <!-- Audio Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('audio')">
                    <h3>Audio Interface</h3>
                    <span class="collapse-icon">{{ collapsedGroups.audio ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.audio">
                    <div v-if="!audioInitialized" class="audio-init-warning">
                        <p>Audio playback requires initialization</p>
                        <button @click="initializeAudio" class="audio-init-button">
                            Enable Audio
                        </button>
                    </div>
                    <div v-else class="audio-status">
                        <span class="status-indicator enabled">Audio Enabled</span>
                    </div>
                </div>
            </div>

            <!-- Node Appearance -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('nodeAppearance')">
                    <h3>Node Appearance</h3>
                    <span class="collapse-icon">{{ collapsedGroups.nodeAppearance ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.nodeAppearance">
                    <!-- Base Colors -->
                    <div class="sub-group">
                        <h4>Base Colors</h4>
                        <div class="control-item" v-for="color in nodeColors" :key="color.name">
                            <label>{{ color.label }}</label>
                            <input type="color" 
                                   v-model="color.value" 
                                   @input="emitChange(color.name, color.value)">
                        </div>
                    </div>
                    
                    <!-- Material Properties -->
                    <div class="sub-group">
                        <h4>Material Properties</h4>
                        <div class="control-item" v-for="prop in materialProperties" :key="prop.name">
                            <label>{{ prop.label }}</label>
                            <input type="range"
                                   v-model.number="prop.value"
                                   :min="prop.min"
                                   :max="prop.max"
                                   :step="prop.step"
                                   @input="emitChange(prop.name, prop.value)">
                            <span class="range-value">{{ prop.value }}</span>
                        </div>
                    </div>

                    <!-- Size Controls -->
                    <div class="sub-group">
                        <h4>Size Settings</h4>
                        <div class="control-item" v-for="size in sizeControls" :key="size.name">
                            <label>{{ size.label }}</label>
                            <input type="range"
                                   v-model.number="size.value"
                                   :min="size.min"
                                   :max="size.max"
                                   :step="size.step"
                                   @input="emitChange(size.name, size.value)">
                            <span class="range-value">{{ size.value }}m</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Edge Appearance -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('edgeAppearance')">
                    <h3>Edge Appearance</h3>
                    <span class="collapse-icon">{{ collapsedGroups.edgeAppearance ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.edgeAppearance">
                    <div class="sub-group">
                        <div class="control-item" v-for="control in edgeControls" :key="control.name">
                            <label>{{ control.label }}</label>
                            <template v-if="control.type === 'color'">
                                <input type="color"
                                       v-model="control.value"
                                       @input="emitChange(control.name, control.value)">
                            </template>
                            <template v-else>
                                <input type="range"
                                       v-model.number="control.value"
                                       :min="control.min"
                                       :max="control.max"
                                       :step="control.step"
                                       @input="emitChange(control.name, control.value)">
                                <span class="range-value">{{ control.value }}</span>
                            </template>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bloom Effects -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('bloom')">
                    <h3>Bloom Effects</h3>
                    <span class="collapse-icon">{{ collapsedGroups.bloom ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.bloom">
                    <div class="sub-group" v-for="(group, key) in bloomControls" :key="key">
                        <h4>{{ group.label }}</h4>
                        <div class="control-item" v-for="control in group.controls" :key="control.name">
                            <label>{{ control.label }}</label>
                            <input type="range"
                                   v-model.number="control.value"
                                   :min="control.min"
                                   :max="control.max"
                                   :step="control.step"
                                   @input="emitChange(control.name, control.value)">
                            <span class="range-value">{{ control.value }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Physics Simulation -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('physics')">
                    <h3>Physics Simulation</h3>
                    <span class="collapse-icon">{{ collapsedGroups.physics ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.physics">
                    <div class="control-item">
                        <label>Simulation Mode</label>
                        <select v-model="simulationMode" @change="setSimulationMode">
                            <option value="remote">Remote (GPU Server)</option>
                            <option value="gpu">Local GPU</option>
                            <option value="local">Local CPU</option>
                        </select>
                    </div>
                    <div class="control-item" v-for="control in physicsControls" :key="control.name">
                        <label>{{ control.label }}</label>
                        <input type="range"
                               v-model.number="control.value"
                               :min="control.min"
                               :max="control.max"
                               :step="control.step"
                               @input="emitChange(control.name, control.value)">
                        <span class="range-value">{{ control.value }}</span>
                    </div>
                </div>
            </div>

            <!-- Environment Settings -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('environment')">
                    <h3>Environment</h3>
                    <span class="collapse-icon">{{ collapsedGroups.environment ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.environment">
                    <div class="sub-group">
                        <h4>Hologram Settings</h4>
                        <div class="control-item" v-for="control in hologramControls" :key="control.name">
                            <label>{{ control.label }}</label>
                            <template v-if="control.type === 'color'">
                                <input type="color"
                                       v-model="control.value"
                                       @input="emitChange(control.name, control.value)">
                            </template>
                            <template v-else>
                                <input type="range"
                                       v-model.number="control.value"
                                       :min="control.min"
                                       :max="control.max"
                                       :step="control.step"
                                       @input="emitChange(control.name, control.value)">
                                <span class="range-value">{{ control.value }}</span>
                            </template>
                        </div>
                    </div>
                    <div class="sub-group">
                        <h4>Fog Settings</h4>
                        <div class="control-item">
                            <label>Fog Density</label>
                            <input type="range"
                                   v-model.number="fogDensity"
                                   min="0"
                                   max="0.01"
                                   step="0.0001"
                                   @input="emitChange('fogDensity', fogDensity)">
                            <span class="range-value">{{ fogDensity }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Fisheye Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('fisheye')">
                    <h3>Fisheye Effect</h3>
                    <span class="collapse-icon">{{ collapsedGroups.fisheye ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.fisheye">
                    <div class="control-item">
                        <label>Enable Fisheye</label>
                        <input type="checkbox"
                               v-model="fisheyeEnabled"
                               @change="emitChange('fisheyeEnabled', fisheyeEnabled)">
                    </div>
                    <div class="control-item" v-for="control in fisheyeControls" :key="control.name">
                        <label>{{ control.label }}</label>
                        <input type="range"
                               v-model.number="control.value"
                               :min="control.min"
                               :max="control.max"
                               :step="control.step"
                               :disabled="!fisheyeEnabled"
                               @input="emitChange(control.name, control.value)">
                        <span class="range-value">{{ control.value }}</span>
                    </div>
                </div>
            </div>

            <!-- Save Settings Button -->
            <button @click="saveSettings" class="save-button">Save Settings</button>
        </div>
    </div>
</template>

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
                        node: {
                            colors: nodeColors.reduce((acc, color) => ({
                                ...acc,
                                [color.name]: color.value
                            }), {}),
                            materialMetalness: nodeMaterialMetalness.value,
                            materialRoughness: nodeMaterialRoughness.value,
                            materialClearcoat: nodeMaterialClearcoat.value,
                            materialClearcoatRoughness: nodeMaterialClearcoatRoughness.value,
                            materialOpacity: nodeMaterialOpacity.value,
                            emissiveMin: nodeEmissiveMin.value,
                            emissiveMax: nodeEmissiveMax.value
                        },
                        edge: {
                            color: edgeColor.value,
                            opacity: edgeOpacity.value,
                            weightNorm: edgeWeightNorm.value,
                            minWidth: edgeMinWidth.value,
                            maxWidth: edgeMaxWidth.value
                        },
                        bloom: {
                            nodes: {
                                strength: nodeBloomStrength.value,
                                radius: nodeBloomRadius.value,
                                threshold: nodeBloomThreshold.value
                            },
                            edges: {
                                strength: edgeBloomStrength.value,
                                radius: edgeBloomRadius.value,
                                threshold: edgeBloomThreshold.value
                            },
                            environment: {
                                strength: envBloomStrength.value,
                                radius: envBloomRadius.value,
                                threshold: envBloomThreshold.value
                            }
                        },
                        physics: {
                            iterations: iterations.value,
                            springStrength: springStrength.value,
                            repulsionStrength: repulsionStrength.value,
                            attractionStrength: attractionStrength.value,
                            damping: damping.value
                        },
                        environment: {
                            hologram: {
                                color: hologramColor.value,
                                scale: hologramScale.value,
                                opacity: hologramOpacity.value
                            },
                            fogDensity: fogDensity.value
                        },
                        fisheye: {
                            enabled: fisheyeEnabled.value,
                            strength: fisheyeStrength.value,
                            radius: fisheyeRadius.value,
                            focusX: fisheyeFocusX.value,
                            focusY: fisheyeFocusY.value,
                            focusZ: fisheyeFocusZ.value
                        }
                    }
                };
                
                props.websocketService.send({
                    type: 'saveSettings',
                    settings: settings
                });
            }
        };

        // Initialize settings from server
        onMounted(() => {
            if (props.websocketService) {
                props.websocketService.on('serverSettings', (settings) => {
                    // Update local controls with server settings
                    nodeColors[0].value = settings.visualization?.nodeColor || nodeColors[0].value;
                    nodeColors[1].value = settings.visualization?.nodeColorNew || nodeColors[1].value;
                    nodeColors[2].value = settings.visualization?.nodeColorRecent || nodeColors[2].value;
                    nodeColors[3].value = settings.visualization?.nodeColorMedium || nodeColors[3].value;
                    nodeColors[4].value = settings.visualization?.nodeColorOld || nodeColors[4].value;
                    nodeColors[5].value = settings.visualization?.nodeColorCore || nodeColors[5].value;
                    nodeColors[6].value = settings.visualization?.nodeColorSecondary || nodeColors[6].value;
                    nodeMaterialMetalness.value = settings.visualization?.nodeMaterialMetalness || nodeMaterialMetalness.value;
                    nodeMaterialRoughness.value = settings.visualization?.nodeMaterialRoughness || nodeMaterialRoughness.value;
                    nodeMaterialClearcoat.value = settings.visualization?.nodeMaterialClearcoat || nodeMaterialClearcoat.value;
                    nodeMaterialClearcoatRoughness.value = settings.visualization?.nodeMaterialClearcoatRoughness || nodeMaterialClearcoatRoughness.value;
                    nodeMaterialOpacity.value = settings.visualization?.nodeMaterialOpacity || nodeMaterialOpacity.value;
                    nodeEmissiveMin.value = settings.visualization?.nodeEmissiveMin || nodeEmissiveMin.value;
                    nodeEmissiveMax.value = settings.visualization?.nodeEmissiveMax || nodeEmissiveMax.value;
                    edgeColor.value = settings.visualization?.edgeColor || edgeColor.value;
                    edgeOpacity.value = settings.visualization?.edgeOpacity || edgeOpacity.value;
                    edgeWeightNorm.value = settings.visualization?.edgeWeightNorm || edgeWeightNorm.value;
                    edgeMinWidth.value = settings.visualization?.edgeMinWidth || edgeMinWidth.value;
                    edgeMaxWidth.value = settings.visualization?.edgeMaxWidth || edgeMaxWidth.value;
                    nodeBloomStrength.value = settings.visualization?.nodeBloomStrength || nodeBloomStrength.value;
                    nodeBloomRadius.value = settings.visualization?.nodeBloomRadius || nodeBloomRadius.value;
                    nodeBloomThreshold.value = settings.visualization?.nodeBloomThreshold || nodeBloomThreshold.value;
                    edgeBloomStrength.value = settings.visualization?.edgeBloomStrength || edgeBloomStrength.value;
                    edgeBloomRadius.value = settings.visualization?.edgeBloomRadius || edgeBloomRadius.value;
                    edgeBloomThreshold.value = settings.visualization?.edgeBloomThreshold || edgeBloomThreshold.value;
                    envBloomStrength.value = settings.visualization?.envBloomStrength || envBloomStrength.value;
                    envBloomRadius.value = settings.visualization?.envBloomRadius || envBloomRadius.value;
                    envBloomThreshold.value = settings.visualization?.envBloomThreshold || envBloomThreshold.value;
                    iterations.value = settings.visualization?.iterations || iterations.value;
                    springStrength.value = settings.visualization?.springStrength || springStrength.value;
                    repulsionStrength.value = settings.visualization?.repulsionStrength || repulsionStrength.value;
                    attractionStrength.value = settings.visualization?.attractionStrength || attractionStrength.value;
                    damping.value = settings.visualization?.damping || damping.value;
                    hologramColor.value = settings.visualization?.hologramColor || hologramColor.value;
                    hologramScale.value = settings.visualization?.hologramScale || hologramScale.value;
                    hologramOpacity.value = settings.visualization?.hologramOpacity || hologramOpacity.value;
                    fogDensity.value = settings.visualization?.fogDensity || fogDensity.value;
                    fisheyeEnabled.value = settings.visualization?.fisheyeEnabled || fisheyeEnabled.value;
                    fisheyeStrength.value = settings.visualization?.fisheyeStrength || fisheyeStrength.value;
                    fisheyeRadius.value = settings.visualization?.fisheyeRadius || fisheyeRadius.value;
                    fisheyeFocusX.value = settings.visualization?.fisheyeFocusX || fisheyeFocusX.value;
                    fisheyeFocusY.value = settings.visualization?.fisheyeFocusY || fisheyeFocusY.value;
                    fisheyeFocusZ.value = settings.visualization?.fisheyeFocusZ || fisheyeFocusZ.value;
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
