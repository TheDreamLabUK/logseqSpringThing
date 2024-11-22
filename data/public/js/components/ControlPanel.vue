<script>
import { defineComponent, ref, reactive, onMounted, onBeforeUnmount, watch } from 'vue';

export default defineComponent({
    name: 'ControlPanel',
    props: {
        websocketService: {
            type: Object,
            required: true
        }
    },
    setup(props, { emit }) {
        console.log('ControlPanel setup called');
        
        // Panel visibility state
        const isHidden = ref(false);
        const collapsedGroups = reactive({
            chat: false,
            nodeAppearance: false,
            edgeAppearance: false,
            bloom: false,
            physics: false,
            fisheye: false,
            environment: false,
            additional: false
        });

        // Chat state
        const chatInput = ref('');
        const chatMessages = ref([]);
        const audioInitialized = ref(false);

        // Node Appearance Controls
        const nodeAppearanceControls = reactive({
            colors: [
                { name: 'nodeColor', label: 'Base Color', value: '#FFA500' },
                { name: 'nodeColorNew', label: 'New Nodes', value: '#FFD700' },
                { name: 'nodeColorRecent', label: 'Recent Nodes', value: '#FFA500' },
                { name: 'nodeColorMedium', label: 'Medium-age Nodes', value: '#DAA520' },
                { name: 'nodeColorOld', label: 'Old Nodes', value: '#CD853F' },
                { name: 'nodeColorCore', label: 'Core Nodes', value: '#FFB90F' },
                { name: 'nodeColorSecondary', label: 'Secondary Nodes', value: '#FFC125' }
            ],
            size: [
                { name: 'minNodeSize', label: 'Min Node Size (cm)', value: 15, min: 5, max: 30, step: 1 },
                { name: 'maxNodeSize', label: 'Max Node Size (cm)', value: 40, min: 20, max: 60, step: 1 }
            ],
            material: [
                { name: 'nodeMaterialMetalness', label: 'Metalness', value: 0.7, min: 0, max: 1, step: 0.1 },
                { name: 'nodeMaterialRoughness', label: 'Roughness', value: 0.1, min: 0, max: 1, step: 0.1 },
                { name: 'nodeMaterialClearcoat', label: 'Clearcoat', value: 0.8, min: 0, max: 1, step: 0.1 },
                { name: 'nodeMaterialClearcoatRoughness', label: 'Clearcoat Roughness', value: 0.1, min: 0, max: 1, step: 0.1 },
                { name: 'nodeMaterialOpacity', label: 'Opacity', value: 0.95, min: 0, max: 1, step: 0.05 }
            ],
            emissive: [
                { name: 'nodeEmissiveMin', label: 'Min Glow', value: 0.4, min: 0, max: 1, step: 0.1 },
                { name: 'nodeEmissiveMax', label: 'Max Glow', value: 1.2, min: 0, max: 2, step: 0.1 }
            ]
        });

        // Edge Appearance Controls
        const edgeAppearanceControls = reactive({
            colors: [
                { name: 'edgeColor', label: 'Edge Color', value: '#FFD700' }
            ],
            properties: [
                { name: 'edgeOpacity', label: 'Edge Opacity', value: 0.4, min: 0, max: 1, step: 0.1 },
                { name: 'edgeWeightNorm', label: 'Weight Normalization', value: 12, min: 1, max: 20, step: 1 },
                { name: 'edgeMinWidth', label: 'Min Width', value: 1.5, min: 0.5, max: 5, step: 0.5 },
                { name: 'edgeMaxWidth', label: 'Max Width', value: 6.0, min: 2, max: 10, step: 0.5 }
            ]
        });

        // Bloom Effect Controls
        const bloomControls = reactive({
            node: [
                { name: 'nodeBloomStrength', label: 'Node Strength', value: 0.8, min: 0, max: 2, step: 0.1 },
                { name: 'nodeBloomRadius', label: 'Node Radius', value: 0.3, min: 0, max: 1, step: 0.1 },
                { name: 'nodeBloomThreshold', label: 'Node Threshold', value: 0.2, min: 0, max: 1, step: 0.1 }
            ],
            edge: [
                { name: 'edgeBloomStrength', label: 'Edge Strength', value: 0.6, min: 0, max: 2, step: 0.1 },
                { name: 'edgeBloomRadius', label: 'Edge Radius', value: 0.4, min: 0, max: 1, step: 0.1 },
                { name: 'edgeBloomThreshold', label: 'Edge Threshold', value: 0.1, min: 0, max: 1, step: 0.1 }
            ],
            environment: [
                { name: 'envBloomStrength', label: 'Environment Strength', value: 0.7, min: 0, max: 2, step: 0.1 },
                { name: 'envBloomRadius', label: 'Environment Radius', value: 0.3, min: 0, max: 1, step: 0.1 },
                { name: 'envBloomThreshold', label: 'Environment Threshold', value: 0.1, min: 0, max: 1, step: 0.1 }
            ]
        });

        // Physics Simulation Controls
        const physicsControls = reactive({
            simulation: [
                { name: 'forceDirectedIterations', label: 'Iterations', value: 300, min: 100, max: 500, step: 10 },
                { name: 'forceDirectedSpring', label: 'Spring Strength', value: 0.015, min: 0.001, max: 0.1, step: 0.001 },
                { name: 'forceDirectedRepulsion', label: 'Repulsion', value: 1200, min: 100, max: 5000, step: 100 },
                { name: 'forceDirectedAttraction', label: 'Attraction', value: 0.012, min: 0.001, max: 0.1, step: 0.001 },
                { name: 'forceDirectedDamping', label: 'Damping', value: 0.85, min: 0.1, max: 0.99, step: 0.01 }
            ],
            geometry: [
                { name: 'geometryMinSegments', label: 'Min Segments', value: 24, min: 8, max: 32, step: 4 },
                { name: 'geometryMaxSegments', label: 'Max Segments', value: 48, min: 24, max: 64, step: 4 },
                { name: 'geometrySegmentPerLink', label: 'Segments per Link', value: 0.6, min: 0.2, max: 1, step: 0.1 }
            ]
        });

        // Fisheye Controls
        const fisheyeEnabled = ref(false);
        const fisheyeControls = reactive({
            properties: [
                { name: 'fisheyeStrength', label: 'Strength', value: 0.5, min: 0, max: 1, step: 0.1 },
                { name: 'fisheyeRadius', label: 'Radius', value: 100, min: 10, max: 200, step: 10 }
            ],
            focus: [
                { name: 'fisheyeFocusX', label: 'Focus X', value: 0, min: -100, max: 100, step: 1 },
                { name: 'fisheyeFocusY', label: 'Focus Y', value: 0, min: -100, max: 100, step: 1 },
                { name: 'fisheyeFocusZ', label: 'Focus Z', value: 0, min: -100, max: 100, step: 1 }
            ]
        });

        // Environment Controls
        const environmentControls = reactive({
            hologram: [
                { name: 'hologramColor', label: 'Hologram Color', value: '#FFC125' },
                { name: 'hologramScale', label: 'Scale', value: 6.0, min: 1, max: 10, step: 0.5 },
                { name: 'hologramOpacity', label: 'Opacity', value: 0.15, min: 0, max: 1, step: 0.05 }
            ],
            fog: [
                { name: 'fogDensity', label: 'Fog Density', value: 0.001, min: 0, max: 0.01, step: 0.001 }
            ]
        });

        // Simulation mode
        const simulationMode = ref('remote');

        // Methods
        const togglePanel = () => {
            isHidden.value = !isHidden.value;
        };

        const toggleGroup = (group) => {
            collapsedGroups[group] = !collapsedGroups[group];
        };

        const emitChange = (name, value) => {
            console.log('Emitting control change:', name, value);
            emit('control-change', { name, value });
        };

        const saveSettings = () => {
            // Collect all settings into TOML structure
            const settings = {
                visualization: {
                    // Node colors
                    node_color: nodeAppearanceControls.colors.find(c => c.name === 'nodeColor')?.value.replace('#', '0x'),
                    node_color_new: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorNew')?.value.replace('#', '0x'),
                    node_color_recent: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorRecent')?.value.replace('#', '0x'),
                    node_color_medium: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorMedium')?.value.replace('#', '0x'),
                    node_color_old: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorOld')?.value.replace('#', '0x'),
                    node_color_core: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorCore')?.value.replace('#', '0x'),
                    node_color_secondary: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorSecondary')?.value.replace('#', '0x'),

                    // Edge colors
                    edge_color: edgeAppearanceControls.colors.find(c => c.name === 'edgeColor')?.value.replace('#', '0x'),
                    
                    // Hologram
                    hologram_color: environmentControls.hologram.find(c => c.name === 'hologramColor')?.value.replace('#', '0x'),
                    hologram_scale: environmentControls.hologram.find(c => c.name === 'hologramScale')?.value,
                    hologram_opacity: environmentControls.hologram.find(c => c.name === 'hologramOpacity')?.value,

                    // Node dimensions
                    min_node_size: nodeAppearanceControls.size.find(c => c.name === 'minNodeSize')?.value / 100, // Convert cm to meters
                    max_node_size: nodeAppearanceControls.size.find(c => c.name === 'maxNodeSize')?.value / 100, // Convert cm to meters
                    
                    // Edge properties
                    edge_opacity: edgeAppearanceControls.properties.find(c => c.name === 'edgeOpacity')?.value,

                    // Material properties
                    node_material_metalness: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialMetalness')?.value,
                    node_material_roughness: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialRoughness')?.value,
                    node_material_clearcoat: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialClearcoat')?.value,
                    node_material_clearcoat_roughness: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialClearcoatRoughness')?.value,
                    node_material_opacity: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialOpacity')?.value,
                    
                    // Emissive properties
                    node_emissive_min_intensity: nodeAppearanceControls.emissive.find(c => c.name === 'nodeEmissiveMin')?.value,
                    node_emissive_max_intensity: nodeAppearanceControls.emissive.find(c => c.name === 'nodeEmissiveMax')?.value,

                    // Geometry properties
                    geometry_min_segments: physicsControls.geometry.find(c => c.name === 'geometryMinSegments')?.value,
                    geometry_max_segments: physicsControls.geometry.find(c => c.name === 'geometryMaxSegments')?.value,
                    geometry_segment_per_hyperlink: physicsControls.geometry.find(c => c.name === 'geometrySegmentPerLink')?.value,

                    // Physics simulation parameters
                    force_directed_iterations: physicsControls.simulation.find(c => c.name === 'forceDirectedIterations')?.value,
                    force_directed_spring: physicsControls.simulation.find(c => c.name === 'forceDirectedSpring')?.value,
                    force_directed_repulsion: physicsControls.simulation.find(c => c.name === 'forceDirectedRepulsion')?.value,
                    force_directed_attraction: physicsControls.simulation.find(c => c.name === 'forceDirectedAttraction')?.value,
                    force_directed_damping: physicsControls.simulation.find(c => c.name === 'forceDirectedDamping')?.value,

                    // Environment
                    fog_density: environmentControls.fog.find(c => c.name === 'fogDensity')?.value,
                },
                bloom: {
                    // Node bloom
                    node_bloom_strength: bloomControls.node.find(c => c.name === 'nodeBloomStrength')?.value,
                    node_bloom_radius: bloomControls.node.find(c => c.name === 'nodeBloomRadius')?.value,
                    node_bloom_threshold: bloomControls.node.find(c => c.name === 'nodeBloomThreshold')?.value,
                    
                    // Edge bloom
                    edge_bloom_strength: bloomControls.edge.find(c => c.name === 'edgeBloomStrength')?.value,
                    edge_bloom_radius: bloomControls.edge.find(c => c.name === 'edgeBloomRadius')?.value,
                    edge_bloom_threshold: bloomControls.edge.find(c => c.name === 'edgeBloomThreshold')?.value,
                    
                    // Environment bloom
                    environment_bloom_strength: bloomControls.environment.find(c => c.name === 'envBloomStrength')?.value,
                    environment_bloom_radius: bloomControls.environment.find(c => c.name === 'envBloomRadius')?.value,
                    environment_bloom_threshold: bloomControls.environment.find(c => c.name === 'envBloomThreshold')?.value
                },
                fisheye: {
                    enabled: fisheyeEnabled.value,
                    strength: fisheyeControls.properties.find(c => c.name === 'fisheyeStrength')?.value,
                    radius: fisheyeControls.properties.find(c => c.name === 'fisheyeRadius')?.value,
                    focus_x: fisheyeControls.focus.find(c => c.name === 'fisheyeFocusX')?.value,
                    focus_y: fisheyeControls.focus.find(c => c.name === 'fisheyeFocusY')?.value,
                    focus_z: fisheyeControls.focus.find(c => c.name === 'fisheyeFocusZ')?.value
                }
            };

            // Send settings to backend through websocket
            props.websocketService.send({
                type: 'updateSettings',
                settings
            });
        };

        // Return all reactive state and methods
        return {
            isHidden,
            collapsedGroups,
            nodeAppearanceControls,
            edgeAppearanceControls,
            bloomControls,
            physicsControls,
            fisheyeEnabled,
            fisheyeControls,
            environmentControls,
            simulationMode,
            togglePanel,
            toggleGroup,
            emitChange,
            saveSettings
        };
    }
});
</script>

<template>
    <div id="control-panel" :class="{ hidden: isHidden }">
        <button @click="togglePanel" class="toggle-button">
            {{ isHidden ? '>' : '<' }}
        </button>
        <div class="panel-content" v-show="!isHidden">
            <!-- Node Appearance -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('nodeAppearance')">
                    <h3>Node Appearance</h3>
                    <span class="collapse-icon">{{ collapsedGroups.nodeAppearance ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.nodeAppearance">
                    <h4>Colors</h4>
                    <div v-for="control in nodeAppearanceControls.colors" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="color"
                            v-model="control.value"
                            @input="emitChange(control.name, control.value)"
                        >
                    </div>

                    <h4>Size</h4>
                    <div v-for="control in nodeAppearanceControls.size" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>

                    <h4>Material</h4>
                    <div v-for="control in nodeAppearanceControls.material" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>

                    <h4>Emissive</h4>
                    <div v-for="control in nodeAppearanceControls.emissive" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
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
                    <h4>Colors</h4>
                    <div v-for="control in edgeAppearanceControls.colors" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="color"
                            v-model="control.value"
                            @input="emitChange(control.name, control.value)"
                        >
                    </div>

                    <h4>Properties</h4>
                    <div v-for="control in edgeAppearanceControls.properties" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
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
                    <h4>Node Bloom</h4>
                    <div v-for="control in bloomControls.node" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>

                    <h4>Edge Bloom</h4>
                    <div v-for="control in bloomControls.edge" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>

                    <h4>Environment Bloom</h4>
                    <div v-for="control in bloomControls.environment" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
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
                        <select v-model="simulationMode" @change="emitChange('simulationMode', simulationMode)">
                            <option value="remote">Remote (GPU Server)</option>
                            <option value="gpu">Local GPU</option>
                            <option value="local">Local CPU</option>
                        </select>
                    </div>

                    <h4>Force Parameters</h4>
                    <div v-for="control in physicsControls.simulation" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>

                    <h4>Geometry</h4>
                    <div v-for="control in physicsControls.geometry" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
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
                        <div class="toggle-switch">
                            <input
                                type="checkbox"
                                v-model="fisheyeEnabled"
                                @change="emitChange('fisheyeEnabled', fisheyeEnabled)"
                            >
                            <span class="toggle-label">{{ fisheyeEnabled ? 'Enabled' : 'Disabled' }}</span>
                        </div>
                    </div>

                    <h4>Properties</h4>
                    <div v-for="control in fisheyeControls.properties" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>

                    <h4>Focus Point</h4>
                    <div v-for="control in fisheyeControls.focus" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
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
                    <h4>Hologram</h4>
                    <div v-for="control in environmentControls.hologram" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <template v-if="control.name === 'hologramColor'">
                            <input
                                type="color"
                                v-model="control.value"
                                @input="emitChange(control.name, control.value)"
                            >
                        </template>
                        <template v-else>
                            <input
                                type="range"
                                v-model.number="control.value"
                                :min="control.min"
                                :max="control.max"
                                :step="control.step"
                                @input="emitChange(control.name, control.value)"
                            >
                            <span class="range-value">{{ control.value }}</span>
                        </template>
                    </div>

                    <h4>Fog</h4>
                    <div v-for="control in environmentControls.fog" :key="control.name" class="control-item">
                        <label>{{ control.label }}</label>
                        <input
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>
                </div>
            </div>

            <!-- Settings Actions -->
            <div class="settings-actions">
                <button @click="saveSettings" class="save-button">Save Settings to File</button>
            </div>
        </div>
    </div>
</template>

<style scoped>
#control-panel {
    position: fixed;
    top: 20px;
    right: 0;
    width: 300px;
    max-height: 90vh;
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    border-radius: 10px 0 0 10px;
    overflow-y: auto;
    z-index: 1000;
    transition: transform 0.3s ease-in-out;
    display: block !important;
    visibility: visible !important;
}

#control-panel.hidden {
    transform: translateX(calc(100% - 40px));
}

.toggle-button {
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    border: none;
    padding: 10px;
    cursor: pointer;
    border-radius: 5px 0 0 5px;
    z-index: 1001;
    display: block !important;
    visibility: visible !important;
}

.panel-content {
    padding: 20px 20px 20px 40px;
    height: 100%;
    overflow-y: auto;
}

.control-group {
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    overflow: hidden;
}

.group-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.1);
    cursor: pointer;
}

.group-header h3 {
    margin: 0;
    font-size: 1em;
}

.group-content {
    padding: 10px;
}

.control-item {
    margin-bottom: 15px;
}

.control-item label {
    display: block;
    margin-bottom: 5px;
}

.control-item input[type="range"] {
    width: 100%;
    margin-bottom: 5px;
}

.control-item input[type="color"] {
    width: 100%;
    height: 30px;
    padding: 0;
    border: none;
    border-radius: 4px;
}

.range-value {
    float: right;
    font-size: 0.9em;
    color: #aaa;
}

.button-group {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
}

.control-button {
    flex: 1;
    padding: 8px;
    background-color: #444;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.control-button:hover {
    background-color: #555;
}

.reset-button {
    width: 100%;
    padding: 10px;
    background-color: #d32f2f;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 10px;
}

.reset-button:hover {
    background-color: #b71c1c;
}

.chat-messages {
    max-height: 200px;
    overflow-y: auto;
    margin-bottom: 10px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

.message {
    margin-bottom: 10px;
    padding: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

.save-button {
    width: 100%;
    padding: 12px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    font-size: 1.1em;
    transition: background-color 0.2s;
}

.save-button:hover {
    background-color: #45a049;
}

/* Previous styles remain the same */
</style>
