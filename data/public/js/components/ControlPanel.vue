<!-- ControlPanel.vue -->
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
            fisheye: false,
            colors: false,
            sizeOpacity: false,
            bloom: false,
            forceDirected: false,
            additional: false
        });

        // Chat state
        const chatInput = ref('');
        const chatMessages = ref([]);
        const audioInitialized = ref(false);

        // Fisheye controls state
        const fisheyeEnabled = ref(false);
        const fisheyeStrength = ref(0.5);
        const fisheyeRadius = ref(100);
        const fisheyeFocusPoint = ref([0, 0, 0]);

        // Color controls
        const colorControls = reactive([
            { name: 'nodeColor', label: 'Node Color', value: '#ffffff' },
            { name: 'edgeColor', label: 'Edge Color', value: '#888888' },
            { name: 'hologramColor', label: 'Hologram Color', value: '#00ff00' }
        ]);

        // Size and opacity controls
        const sizeOpacityControls = reactive([
            { name: 'nodeSize', label: 'Node Size', value: 1, min: 0.1, max: 5, step: 0.1 },
            { name: 'edgeSize', label: 'Edge Size', value: 1, min: 0.1, max: 5, step: 0.1 },
            { name: 'nodeOpacity', label: 'Node Opacity', value: 1, min: 0, max: 1, step: 0.1 },
            { name: 'edgeOpacity', label: 'Edge Opacity', value: 1, min: 0, max: 1, step: 0.1 }
        ]);

        // Bloom controls
        const bloomControls = reactive([
            { name: 'bloomStrength', label: 'Bloom Strength', value: 1.5, min: 0, max: 3, step: 0.1 },
            { name: 'bloomRadius', label: 'Bloom Radius', value: 0.4, min: 0, max: 1, step: 0.1 },
            { name: 'bloomThreshold', label: 'Bloom Threshold', value: 0.8, min: 0, max: 1, step: 0.1 }
        ]);

        // Force-directed controls
        const forceDirectedControls = reactive([
            { name: 'force_directed_iterations', label: 'Iterations', value: 250, min: 50, max: 500, step: 10 },
            { name: 'force_directed_spring', label: 'Spring Strength', value: 0.1, min: 0.01, max: 1, step: 0.01 },
            { name: 'force_directed_repulsion', label: 'Repulsion', value: 1000, min: 100, max: 5000, step: 100 },
            { name: 'force_directed_attraction', label: 'Attraction', value: 0.01, min: 0.001, max: 0.1, step: 0.001 }
        ]);

        // Additional controls
        const additionalControls = reactive([
            { name: 'cameraSpeed', label: 'Camera Speed', value: 1, min: 0.1, max: 5, step: 0.1 },
            { name: 'zoomSpeed', label: 'Zoom Speed', value: 1, min: 0.1, max: 5, step: 0.1 }
        ]);

        // Simulation mode
        const simulationMode = ref('remote');
        const damping = ref(0.8);

        // XR state
        const xrActive = ref(false);
        const xrSupported = ref(false);
        const xrMode = ref('vr');

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

        const initializeAudio = () => {
            if (props.websocketService) {
                props.websocketService.initAudio();
                audioInitialized.value = true;
            }
        };

        const sendMessage = () => {
            if (chatInput.value.trim()) {
                const message = {
                    message: chatInput.value,
                    timestamp: new Date().toLocaleTimeString()
                };
                chatMessages.value.push(message);
                props.websocketService.sendChatMessage({
                    message: chatInput.value,
                    useOpenAI: false
                });
                chatInput.value = '';
            }
        };

        const setSimulationMode = () => {
            props.websocketService.setSimulationMode(simulationMode.value);
        };

        const toggleFullscreen = () => {
            emit('toggle-fullscreen');
        };

        const enableSpacemouse = () => {
            emit('enable-spacemouse');
        };

        const resetControls = () => {
            // Reset all controls to their default values
            fisheyeEnabled.value = false;
            fisheyeStrength.value = 0.5;
            fisheyeRadius.value = 100;
            fisheyeFocusPoint.value = [0, 0, 0];
            
            colorControls.forEach(control => {
                control.value = control.name === 'nodeColor' ? '#ffffff' :
                              control.name === 'edgeColor' ? '#888888' : '#00ff00';
            });

            sizeOpacityControls.forEach(control => {
                control.value = 1;
            });

            bloomControls.forEach(control => {
                control.value = control.name === 'bloomStrength' ? 1.5 :
                              control.name === 'bloomRadius' ? 0.4 : 0.8;
            });

            forceDirectedControls.forEach(control => {
                control.value = control.name === 'force_directed_iterations' ? 250 :
                              control.name === 'force_directed_spring' ? 0.1 :
                              control.name === 'force_directed_repulsion' ? 1000 : 0.01;
            });

            damping.value = 0.8;
            simulationMode.value = 'remote';

            // Emit all the reset values
            emitChange('fisheyeEnabled', false);
            emitChange('fisheyeStrength', 0.5);
            emitChange('fisheyeRadius', 100);
            emitChange('fisheyeFocusPoint', [0, 0, 0]);
            colorControls.forEach(control => emitChange(control.name, control.value));
            sizeOpacityControls.forEach(control => emitChange(control.name, control.value));
            bloomControls.forEach(control => emitChange(control.name, control.value));
            forceDirectedControls.forEach(control => emitChange(control.name, control.value));
            emitChange('force_directed_damping', 0.8);
            setSimulationMode();
        };

        // XR methods
        const checkXRSupport = async () => {
            if ('xr' in navigator) {
                try {
                    const vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
                    const arSupported = await navigator.xr.isSessionSupported('immersive-ar');
                    xrSupported.value = vrSupported || arSupported;
                    
                    if (arSupported) {
                        xrMode.value = 'ar';
                    } else if (vrSupported) {
                        xrMode.value = 'vr';
                    }
                } catch (err) {
                    console.error('Error checking XR support:', err);
                    xrSupported.value = false;
                }
            } else {
                console.warn('WebXR not supported in this browser');
                xrSupported.value = false;
            }
        };

        const toggleXR = () => {
            xrActive.value = !xrActive.value;
            emit('control-change', { 
                name: 'xrEnabled', 
                value: { 
                    active: xrActive.value, 
                    mode: xrMode.value 
                }
            });
        };

        const changeXRMode = () => {
            emit('control-change', { 
                name: 'xrEnabled', 
                value: { 
                    active: xrActive.value, 
                    mode: xrMode.value 
                }
            });
        };

        // Lifecycle hooks
        onMounted(() => {
            console.log('ControlPanel mounted');
            // Force visibility after mount
            const panel = document.getElementById('control-panel');
            if (panel) {
                panel.style.display = 'block';
                panel.style.visibility = 'visible';
            }

            // Check XR support
            checkXRSupport();

            // Listen for XR session state changes
            window.addEventListener('xrsessionstart', () => {
                xrActive.value = true;
            });

            window.addEventListener('xrsessionend', () => {
                xrActive.value = false;
            });
        });

        onBeforeUnmount(() => {
            window.removeEventListener('xrsessionstart', () => {
                xrActive.value = true;
            });
            window.removeEventListener('xrsessionend', () => {
                xrActive.value = false;
            });
        });

        // Return all reactive state and methods
        return {
            isHidden,
            collapsedGroups,
            chatInput,
            chatMessages,
            audioInitialized,
            fisheyeEnabled,
            fisheyeStrength,
            fisheyeRadius,
            fisheyeFocusPoint,
            colorControls,
            sizeOpacityControls,
            bloomControls,
            forceDirectedControls,
            additionalControls,
            simulationMode,
            damping,
            xrActive,
            xrSupported,
            xrMode,
            togglePanel,
            toggleGroup,
            emitChange,
            initializeAudio,
            sendMessage,
            setSimulationMode,
            toggleFullscreen,
            enableSpacemouse,
            resetControls,
            toggleXR,
            changeXRMode
        };
    }
});
</script>

<template>
    <div id="control-panel" :class="{ hidden: isHidden }" style="display: block !important;">
        <button @click="togglePanel" class="toggle-button" style="display: block !important;">
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
                    <div class="chat-messages">
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
                        <label>Enable Fisheye</label>
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
                        <label>Fisheye Strength</label>
                        <input
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
                        <label>Fisheye Radius</label>
                        <input
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
                        <label>{{ control.label }}</label>
                        <input
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

            <!-- Bloom Effect Controls -->
            <div class="control-group">
                <div class="group-header" @click="toggleGroup('bloom')">
                    <h3>Bloom Effects</h3>
                    <span class="collapse-icon">{{ collapsedGroups.bloom ? '▼' : '▲' }}</span>
                </div>
                <div class="group-content" v-show="!collapsedGroups.bloom">
                    <div v-for="control in bloomControls" :key="control.name" class="control-item">
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
                    <!-- XR Controls -->
                    <div class="control-item xr-controls" v-if="xrSupported">
                        <label>XR Mode</label>
                        <div class="xr-mode-selector">
                            <select v-model="xrMode" @change="changeXRMode" :disabled="xrActive">
                                <option value="vr">Virtual Reality</option>
                                <option value="ar">Augmented Reality</option>
                            </select>
                        </div>
                        <button 
                            @click="toggleXR" 
                            class="xr-toggle-button"
                            :class="{ active: xrActive }"
                        >
                            {{ xrActive ? 'Exit XR' : 'Enter XR' }}
                        </button>
                    </div>
                    <div v-else class="xr-unsupported">
                        WebXR not supported in this browser
                    </div>

                    <!-- Additional Controls -->
                    <div v-for="control in additionalControls" :key="control.name" class="control-item">
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

.message-header {
    font-size: 0.8em;
    color: #aaa;
    margin-bottom: 4px;
}

.chat-input {
    display: flex;
    gap: 10px;
}

.chat-input input {
    flex: 1;
    padding: 8px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: white;
}

.chat-input button {
    padding: 8px 15px;
    background: #2196f3;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.chat-input button:hover {
    background: #1976d2;
}

.chat-input button.disabled {
    background: #666;
    cursor: not-allowed;
}

.audio-init-warning {
    padding: 10px;
    background: rgba(255, 193, 7, 0.2);
    border: 1px solid rgba(255, 193, 7, 0.5);
    border-radius: 4px;
    margin-bottom: 10px;
}

.audio-init-button {
    width: 100%;
    padding: 8px;
    background: #ffc107;
    color: black;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 5px;
}

.audio-init-button:hover {
    background: #ffb300;
}

.xr-controls {
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;
    background: rgba(0, 0, 0, 0.3);
}

.xr-mode-selector {
    margin-bottom: 10px;
}

.xr-mode-selector select {
    width: 100%;
    padding: 8px;
    background: #333;
    color: white;
    border: 1px solid #444;
    border-radius: 4px;
}

.xr-toggle-button {
    width: 100%;
    padding: 10px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.2s;
}

.xr-toggle-button:hover {
    background: #0056b3;
}

.xr-toggle-button.active {
    background: #dc3545;
}

.xr-toggle-button.active:hover {
    background: #c82333;
}

.xr-unsupported {
    padding: 10px;
    background: rgba(220, 53, 69, 0.2);
    border: 1px solid rgba(220, 53, 69, 0.5);
    color: #dc3545;
    border-radius: 4px;
    text-align: center;
    margin-bottom: 20px;
}

.xr-mode-selector select:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

/* Collapse icon animation */
.collapse-icon {
    transition: transform 0.3s ease;
}

[v-show="false"] + .collapse-icon {
    transform: rotate(180deg);
}
</style>
