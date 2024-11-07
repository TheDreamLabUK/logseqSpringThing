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
        // Reactive state variables
        const isHidden = ref(false);
        const simulationMode = ref('remote');
        const chatMessagesRef = ref(null);

        // Fisheye controls
        const fisheyeEnabled = ref(false);
        const fisheyeStrength = ref(0.5);
        const fisheyeFocusPoint = ref([0, 0, 0]);
        const fisheyeRadius = ref(100.0);

        // Chat controls
        const chatInput = ref('');
        const chatMessages = ref([]);
        const audioInitialized = ref(false);

        // Visualization controls
        const colorControls = ref([]);
        const sizeOpacityControls = ref([]);
        const bloomControls = ref([]);
        const forceDirectedControls = ref([]);
        const additionalControls = ref([]);
        const springConstant = ref(0.5);
        const damping = ref(0.5);

        // UI state - all groups start collapsed
        const collapsedGroups = reactive({
            chat: true,
            fisheye: true,
            colors: true,
            sizeOpacity: true,
            bloom: true,
            forceDirected: true,
            additional: true
        });

        // Event handlers declared in outer scope
        let handleServerSettings;
        let handleFisheyeSettings;

        // Methods
        const togglePanel = () => {
            isHidden.value = !isHidden.value;
        };

        const toggleGroup = (group) => {
            collapsedGroups[group] = !collapsedGroups[group];
        };

        const setSimulationMode = () => {
            console.log('Setting simulation mode to:', simulationMode.value);
            if (props.websocketService) {
                props.websocketService.setSimulationMode(simulationMode.value);
            } else {
                console.error('WebSocketService is undefined');
            }
        };

        const isColorControl = (name) => {
            return colorControls.value.some(control => control.name === name);
        };

        const emitChange = (name, value) => {
            console.log(`Control change: ${name} = ${value}`);

            if (isColorControl(name)) {
                // Convert from #RRGGBB to decimal format
                value = parseInt(value.replace(/^#/, ''), 16);
            }

            if (name.startsWith('fisheye')) {
                const settings = {
                    enabled: fisheyeEnabled.value,
                    strength: fisheyeStrength.value,
                    focusPoint: fisheyeFocusPoint.value,
                    radius: fisheyeRadius.value
                };

                switch (name) {
                    case 'fisheyeEnabled':
                        fisheyeEnabled.value = Boolean(value);
                        settings.enabled = fisheyeEnabled.value;
                        break;
                    case 'fisheyeStrength':
                        fisheyeStrength.value = value;
                        settings.strength = value;
                        break;
                    case 'fisheyeRadius':
                        fisheyeRadius.value = value;
                        settings.radius = value;
                        break;
                    case 'fisheyeFocusPoint':
                        fisheyeFocusPoint.value = value;
                        settings.focusPoint = value;
                        break;
                }

                if (props.websocketService) {
                    console.log('Sending fisheye settings to server:', settings);
                    props.websocketService.updateFisheyeSettings(settings);
                }
                return;
            }

            emit('control-change', { name, value });
        };

        const resetControls = () => {
            fisheyeEnabled.value = false;
            fisheyeStrength.value = 0.5;
            fisheyeRadius.value = 100.0;
            fisheyeFocusPoint.value = [0, 0, 0];

            if (props.websocketService) {
                props.websocketService.updateFisheyeSettings({
                    enabled: false,
                    strength: 0.5,
                    focusPoint: [0, 0, 0],
                    radius: 100.0
                });
            }

            window.dispatchEvent(new CustomEvent('resetVisualizationSettings'));
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

        const sendMessage = () => {
            if (!audioInitialized.value) {
                console.warn('Audio not initialized. Please enable audio first.');
                return;
            }

            if (chatInput.value.trim() && props.websocketService) {
                chatMessages.value.push({
                    sender: 'You',
                    message: chatInput.value,
                    timestamp: new Date().toLocaleTimeString()
                });

                props.websocketService.sendChatMessage({
                    message: chatInput.value,
                    useOpenAI: true
                });

                chatInput.value = '';
            }
        };

        const toggleFullscreen = () => {
            emit('toggle-fullscreen');
        };

        const enableSpacemouse = () => {
            emit('enable-spacemouse');
        };

        const initializeControls = (settings) => {
            console.log('Initializing controls with settings:', settings);

            // Initialize fisheye controls with received values
            if (settings.fisheye) {
                fisheyeEnabled.value = Boolean(settings.fisheye.enabled);
                fisheyeStrength.value = settings.fisheye.strength || 0.5;
                fisheyeFocusPoint.value = settings.fisheye.focusPoint || [0, 0, 0];
                fisheyeRadius.value = settings.fisheye.radius || 100.0;
            }

            // Helper function to convert color values to hex format
            const toHexColor = (value) => {
                if (typeof value === 'string') {
                    if (value.startsWith('#')) {
                        return value;
                    }
                    if (value.startsWith('0x')) {
                        return '#' + value.slice(2).padStart(6, '0');
                    }
                    return '#' + value.padStart(6, '0');
                }
                return '#' + Math.floor(value || 0).toString(16).padStart(6, '0');
            };

            // Initialize controls with received values
            colorControls.value = [
                { 
                    name: 'nodeColor', 
                    label: 'Node Color', 
                    type: 'color', 
                    value: toHexColor(settings.visualization.nodeColor)
                },
                { 
                    name: 'edgeColor', 
                    label: 'Edge Color', 
                    type: 'color', 
                    value: toHexColor(settings.visualization.edgeColor)
                },
                { 
                    name: 'hologramColor', 
                    label: 'Hologram Color', 
                    type: 'color', 
                    value: toHexColor(settings.visualization.hologramColor)
                }
            ];

            sizeOpacityControls.value = [
                { name: 'nodeSizeScalingFactor', label: 'Node Size Scaling', type: 'range', value: settings.visualization.nodeSizeScalingFactor, min: 1, max: 10, step: 0.1 },
                { name: 'hologramScale', label: 'Hologram Scale', type: 'range', value: settings.visualization.hologramScale, min: 1, max: 10, step: 0.1 },
                { name: 'hologramOpacity', label: 'Hologram Opacity', type: 'range', value: settings.visualization.hologramOpacity, min: 0, max: 1, step: 0.01 },
                { name: 'edgeOpacity', label: 'Edge Opacity', type: 'range', value: settings.visualization.edgeOpacity, min: 0, max: 1, step: 0.01 }
            ];

            bloomControls.value = [
                { name: 'nodeBloomStrength', label: 'Node Bloom Strength', type: 'range', value: settings.bloom.nodeBloomStrength, min: 0, max: 1, step: 0.01 },
                { name: 'nodeBloomRadius', label: 'Node Bloom Radius', type: 'range', value: settings.bloom.nodeBloomRadius, min: 0, max: 1, step: 0.01 },
                { name: 'nodeBloomThreshold', label: 'Node Bloom Threshold', type: 'range', value: settings.bloom.nodeBloomThreshold, min: 0, max: 1, step: 0.01 },
                { name: 'edgeBloomStrength', label: 'Edge Bloom Strength', type: 'range', value: settings.bloom.edgeBloomStrength, min: 0, max: 1, step: 0.01 },
                { name: 'edgeBloomRadius', label: 'Edge Bloom Radius', type: 'range', value: settings.bloom.edgeBloomRadius, min: 0, max: 1, step: 0.01 },
                { name: 'edgeBloomThreshold', label: 'Edge Bloom Threshold', type: 'range', value: settings.bloom.edgeBloomThreshold, min: 0, max: 1, step: 0.01 },
                { name: 'environmentBloomStrength', label: 'Environment Bloom Strength', type: 'range', value: settings.bloom.environmentBloomStrength, min: 0, max: 2, step: 0.01 },
                { name: 'environmentBloomRadius', label: 'Environment Bloom Radius', type: 'range', value: settings.bloom.environmentBloomRadius, min: 0, max: 2, step: 0.01 },
                { name: 'environmentBloomThreshold', label: 'Environment Bloom Threshold', type: 'range', value: settings.bloom.environmentBloomThreshold, min: 0, max: 1, step: 0.01 }
            ];

            forceDirectedControls.value = [
                { name: 'iterations', label: 'Iterations', type: 'range', value: settings.visualization.forceDirectedIterations, min: 10, max: 500, step: 10 },
                { name: 'repulsion_strength', label: 'Repulsion', type: 'range', value: settings.visualization.forceDirectedRepulsion, min: 0.1, max: 10.0, step: 0.1 },
                { name: 'attraction_strength', label: 'Attraction', type: 'range', value: settings.visualization.forceDirectedAttraction, min: 0.001, max: 0.1, step: 0.001 }
            ];

            additionalControls.value = [
                { name: 'labelFontSize', label: 'Label Font Size', type: 'range', value: settings.visualization.labelFontSize, min: 12, max: 72, step: 1 },
                { name: 'fogDensity', label: 'Fog Density', type: 'range', value: settings.visualization.fogDensity, min: 0, max: 0.01, step: 0.0001 }
            ];
        };

        // Lifecycle hooks
        onMounted(() => {
            if (props.websocketService) {
                props.websocketService.on('simulationModeSet', (mode) => {
                    console.log('Simulation mode set to:', mode);
                    simulationMode.value = mode;
                });

                // Define event handlers
                handleServerSettings = (event) => {
                    console.log('Received server settings:', event.detail);
                    initializeControls(event.detail);
                };

                handleFisheyeSettings = (event) => {
                    console.log('Received fisheye settings update:', event.detail);
                    const settings = event.detail;
                    fisheyeEnabled.value = settings.enabled;
                    fisheyeStrength.value = settings.strength;
                    fisheyeFocusPoint.value = settings.focusPoint;
                    fisheyeRadius.value = settings.radius;
                };

                // Add event listeners
                window.addEventListener('serverSettings', handleServerSettings);
                window.addEventListener('fisheyeSettingsUpdated', handleFisheyeSettings);
            } else {
                console.error('WebSocketService is undefined');
            }
        });

        // Clean up event listeners
        onBeforeUnmount(() => {
            if (handleServerSettings) {
                window.removeEventListener('serverSettings', handleServerSettings);
            }
            if (handleFisheyeSettings) {
                window.removeEventListener('fisheyeSettingsUpdated', handleFisheyeSettings);
            }
        });

        const scrollToBottom = () => {
            if (chatMessagesRef.value) {
                chatMessagesRef.value.scrollTop = chatMessagesRef.value.scrollHeight;
            }
        };

        watch(chatMessages, () => {
            scrollToBottom();
        });

        return {
            isHidden,
            simulationMode,
            fisheyeEnabled,
            fisheyeStrength,
            fisheyeFocusPoint,
            fisheyeRadius,
            chatInput,
            chatMessages,
            audioInitialized,
            colorControls,
            sizeOpacityControls,
            bloomControls,
            forceDirectedControls,
            additionalControls,
            springConstant,
            damping,
            collapsedGroups,
            togglePanel,
            toggleGroup,
            setSimulationMode,
            emitChange,
            resetControls,
            initializeAudio,
            sendMessage,
            toggleFullscreen,
            enableSpacemouse,
            chatMessagesRef
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
            <!-- Chat Interface -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('chat')">
                    <h3>Chat Interface</h3>
                    <span class="collapse-icon">{{ collapsedGroups.chat ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.chat">
                    <div v-if="!audioInitialized" class="audio-init-warning">
                        <p>Audio playback requires initialization</p>
                        <button @click="initializeAudio" class="audio-init-button">
                            Enable Audio
                        </button>
                    </div>
                    <div class="chat-messages" ref="chatMessagesRef">
                        <div v-for="(message, index) in chatMessages" :key="index" class="message user-message">
                            <div class="message-header">
                                <span class="message-time">{{ message.timestamp }}</span>
                            </div>
                            <div class="message-content">
                                {{ message.message }}
                            </div>
                        </div>
                    </div>
                    <div class="chat-input">
                        <input 
                            v-model="chatInput" 
                            @keyup.enter="sendMessage" 
                            placeholder="Type your message..."
                            :disabled="!audioInitialized"
                        >
                        <button 
                            @click="sendMessage"
                            :disabled="!audioInitialized"
                            :class="{ 'disabled': !audioInitialized }"
                        >
                            Send
                        </button>
                    </div>
                </div>
            </div>
        
            <!-- Fisheye Distortion Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('fisheye')">
                    <h3>Fisheye Distortion</h3>
                    <span class="collapse-icon">{{ collapsedGroups.fisheye ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.fisheye">
                    <div class="control-item">
                        <label for="fisheye_enabled">Enable Fisheye</label>
                        <div>
                            <label>
                                <input 
                                    type="radio" 
                                    :value="true"
                                    v-model="fisheyeEnabled"
                                    @change="emitChange('fisheyeEnabled', true)"
                                >
                                Enable
                            </label>
                            <label>
                                <input 
                                    type="radio" 
                                    :value="false"
                                    v-model="fisheyeEnabled"
                                    @change="emitChange('fisheyeEnabled', false)"
                                >
                                Disable
                            </label>
                        </div>
                    </div>
                    <div class="control-item">
                        <label for="fisheye_strength">Fisheye Strength</label>
                        <input
                            id="fisheye_strength"
                            type="range"
                            v-model.number="fisheyeStrength"
                            :min="0"
                            :max="1"
                            :step="0.01"
                            @input="emitChange('fisheyeStrength', fisheyeStrength)"
                        >
                        <span class="range-value">{{ fisheyeStrength }}</span>
                    </div>
                    <div class="control-item">
                        <label for="fisheye_radius">Fisheye Radius</label>
                        <input
                            id="fisheye_radius"
                            type="range"
                            v-model.number="fisheyeRadius"
                            :min="10"
                            :max="200"
                            :step="1"
                            @input="emitChange('fisheyeRadius', fisheyeRadius)"
                        >
                        <span class="range-value">{{ fisheyeRadius }}</span>
                    </div>
                    <!-- Focus Point Controls -->
                    <div v-for="(axis, index) in ['X', 'Y', 'Z']" :key="axis" class="control-item">
                        <label>Focus Point {{ axis }}</label>
                        <input
                            type="range"
                            v-model.number="fisheyeFocusPoint[index]"
                            :min="-100"
                            :max="100"
                            :step="1"
                            @input="emitChange('fisheyeFocusPoint', fisheyeFocusPoint)"
                        >
                        <span class="range-value">{{ fisheyeFocusPoint[index] }}</span>
                    </div>
                </div>
            </div>

            <!-- Color Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('colors')">
                    <h3>Colors</h3>
                    <span class="collapse-icon">{{ collapsedGroups.colors ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.colors">
                    <div v-for="control in colorControls" :key="control.name" class="control-item">
                        <label :for="control.name">{{ control.label }}</label>
                        <input
                            :id="control.name"
                            type="color"
                            v-model="control.value"
                            @input="emitChange(control.name, control.value)"
                        >
                    </div>
                </div>
            </div>

            <!-- Size and Opacity Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('sizeOpacity')">
                    <h3>Size and Opacity</h3>
                    <span class="collapse-icon">{{ collapsedGroups.sizeOpacity ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.sizeOpacity">
                    <div v-for="control in sizeOpacityControls" :key="control.name" class="control-item">
                        <label :for="control.name">{{ control.label }}</label>
                        <input
                            :id="control.name"
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

            <!-- Bloom Effect Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('bloom')">
                    <h3>Bloom Effects</h3>
                    <span class="collapse-icon">{{ collapsedGroups.bloom ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.bloom">
                    <div v-for="control in bloomControls" :key="control.name" class="control-item">
                        <label :for="control.name">{{ control.label }}</label>
                        <input
                            :id="control.name"
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

            <!-- Force-Directed Graph Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('forceDirected')">
                    <h3>Force-Directed Graph</h3>
                    <span class="collapse-icon">{{ collapsedGroups.forceDirected ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.forceDirected">
                    <div class="control-item">
                        <label>Simulation Mode</label>
                        <select v-model="simulationMode" @change="setSimulationMode">
                            <option value="remote">Remote (GPU Server)</option>
                            <option value="gpu">Local GPU</option>
                            <option value="local">Local CPU</option>
                        </select>
                    </div>
                    <div v-for="control in forceDirectedControls" :key="control.name" class="control-item">
                        <label :for="control.name">{{ control.label }}</label>
                        <input
                            :id="control.name"
                            type="range"
                            v-model.number="control.value"
                            :min="control.min"
                            :max="control.max"
                            :step="control.step"
                            @input="emitChange(control.name, control.value)"
                        >
                        <span class="range-value">{{ control.value }}</span>
                    </div>
                    <div class="control-item">
                        <label>Spring Constant</label>
                        <input
                            type="range"
                            v-model.number="springConstant"
                            :min="0.01"
                            :max="1.0"
                            :step="0.01"
                            @input="emitChange('force_directed_spring', springConstant)"
                        >
                        <span class="range-value">{{ springConstant }}</span>
                    </div>
                    <div class="control-item">
                        <label>Damping</label>
                        <input
                            type="range"
                            v-model.number="damping"
                            :min="0.1"
                            :max="0.9"
                            :step="0.1"
                            @input="emitChange('force_directed_damping', damping)"
                        >
                        <span class="range-value">{{ damping }}</span>
                    </div>
                </div>
            </div>

            <!-- Additional Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('additional')">
                    <h3>Additional Settings</h3>
                    <span class="collapse-icon">{{ collapsedGroups.additional ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.additional">
                    <div v-for="control in additionalControls" :key="control.name" class="control-item">
                        <label :for="control.name">{{ control.label }}</label>
                        <input
                            :id="control.name"
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

            <!-- Additional Buttons -->
            <div class="button-group">
                <button @click="toggleFullscreen" class="control-button">Toggle Fullscreen</button>
                <button @click="enableSpacemouse" class="control-button">Enable Spacemouse</button>
            </div>

            <button @click="resetControls" class="reset-button">Reset to Defaults</button>
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
    z-index: 2;
}

.panel-content {
    padding: 20px 20px 20px 40px;
    height: 100%;
    overflow-y: auto;
}

.control-group {
    margin-bottom: 20px;
    border: 1px solid #444;
    border-radius: 5px;
    overflow: hidden;
}

.group-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    background-color: #333;
    cursor: pointer;
    user-select: none;
}

.group-header h3 {
    margin: 0;
    font-size: 1.1em;
}

.collapse-icon {
    font-size: 0.8em;
    transition: transform 0.3s ease;
}

.group-content {
    padding: 15px;
    background-color: rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.control-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 10px;
}

label {
    font-weight: bold;
    font-size: 0.9em;
}

input[type="color"] {
    width: 100%;
    height: 30px;
    border: none;
    border-radius: 5px;
    background-color: #444;
}

input[type="range"] {
    width: 100%;
    background-color: #444;
}

.range-value {
    font-size: 0.8em;
    text-align: right;
    color: #aaa;
}

select {
    width: 100%;
    padding: 5px;
    background-color: #333;
    color: white;
    border: 1px solid #444;
    border-radius: 5px;
    margin-bottom: 10px;
}

.button-group {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 20px;
}

.control-button, .reset-button {
    width: 100%;
    padding: 10px;
    background-color: #444;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.control-button:hover, .reset-button:hover {
    background-color: #555;
}

.chat-interface {
    margin-bottom: 15px;
}

.chat-messages {
    max-height: 200px;
    overflow-y: auto;
    padding: 10px;
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    margin-bottom: 10px;
}

.message {
    margin-bottom: 8px;
    padding: 5px;
    border-radius: 3px;
    background-color: rgba(255, 255, 255, 0.1);
}

.user-message {
    margin-left: 20%;
    background-color: rgba(0, 123, 255, 0.2);
}

.message-header {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 4px;
}

.message-time {
    font-size: 0.8em;
    color: rgba(255, 255, 255, 0.6);
}

.message-content {
    word-break: break-word;
}

.chat-input {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.chat-input input[type="text"] {
    width: 100%;
    padding: 8px;
    background-color: #333;
    border: 1px solid #444;
    border-radius: 5px;
    color: white;
}

.audio-init-warning {
    background-color: rgba(255, 193, 7, 0.2);
    border: 1px solid rgba(255, 193, 7, 0.5);
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    text-align: center;
}

.audio-init-warning p {
    margin: 0 0 10px 0;
    color: rgba(255, 193, 7, 0.9);
}

.audio-init-button {
    background-color: rgba(255, 193, 7, 0.8);
    color: black;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.2s;
}

.audio-init-button:hover {
    background-color: rgba(255, 193, 7, 1);
}

.chat-input input:disabled,
.chat-input button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #666;
}
</style>
