<!-- Template section -->
<template>
  <div id="control-panel" :class="{ hidden: isHidden }">
    <button @click="togglePanel" class="toggle-button">
      {{ isHidden ? '>' : '<' }}
    </button>
    <div class="panel-content" v-show="!isHidden">
      <!-- Chat Interface -->
      <div class="chat-interface">
        <div class="chat-messages" ref="chatMessagesRef">
          <div v-for="(message, index) in chatMessages" :key="index" :class="['chat-message', message.sender === 'You' ? 'user' : 'ai']">
            <strong>{{ message.sender }}:</strong> {{ message.message }}
          </div>
        </div>
        <div class="chat-input-container">
          <input type="text" v-model="chatInput" @keyup.enter="sendMessage" placeholder="Type a message..." />
          <div class="chat-controls">
            <label class="tts-toggle">
              <input type="checkbox" v-model="useOpenAITTS">
              Use OpenAI TTS
            </label>
            <button @click="sendMessage">Send</button>
          </div>
        </div>
      </div>

      <!-- Fisheye Distortion Controls -->
      <div class="control-group">
        <h3>Fisheye Distortion</h3>
        <div class="control-item">
          <label for="fisheye_enabled">Enable Fisheye</label>
          <div>
            <label>
              <input type="radio" value="true" v-model="fisheyeEnabled" @change="emitChange('fisheyeEnabled', true)">
              Enable
            </label>
            <label>
              <input type="radio" value="false" v-model="fisheyeEnabled" @change="emitChange('fisheyeEnabled', false)">
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
      </div>

      <!-- Color Controls -->
      <div class="control-group">
        <h3>Colors</h3>
        <div v-for="control in colorControls" :key="control.name" class="control-item">
          <label :for="control.name">{{ control.label }}</label>
          <input
            :id="control.name"
            type="color"
            v-model="control.value"
            @change="emitChange(control.name, control.value)"
          >
        </div>
      </div>

      <!-- Size and Opacity Controls -->
      <div class="control-group">
        <h3>Size and Opacity</h3>
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

      <!-- Bloom Effect Controls -->
      <div class="control-group">
        <h3>Bloom Effects</h3>
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

      <!-- Force-Directed Graph Controls -->
      <div class="control-group">
        <h3>Force-Directed Graph</h3>
        <!-- Add Simulation Mode Toggle -->
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
      </div>

      <!-- Additional Controls -->
      <div class="control-group">
        <h3>Additional Settings</h3>
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

      <!-- Additional Buttons -->
      <div class="button-group">
        <button @click="toggleFullscreen" class="control-button">Toggle Fullscreen</button>
        <button @click="enableSpacemouse" class="control-button">Enable Spacemouse</button>
      </div>

      <button @click="resetControls" class="reset-button">Reset to Defaults</button>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onBeforeUnmount, onUpdated } from 'vue';

export default {
  name: 'ControlPanel',
  props: {
    websocketService: {
      type: Object,
      required: true
    }
  },
  data() {
    return {
      isHidden: false,
      simulationMode: 'remote',
      fisheyeEnabled: false,
      fisheyeStrength: 0.5,
      chatInput: '',
      chatMessages: [],
      useOpenAITTS: false,
      // Initialize controls with empty arrays, will be populated from settings
      colorControls: [],
      sizeOpacityControls: [],
      bloomControls: [],
      forceDirectedControls: [],
      additionalControls: []
    };
  },
  methods: {
    togglePanel() {
      this.isHidden = !this.isHidden;
    },
    setSimulationMode() {
      console.log('Setting simulation mode to:', this.simulationMode);
      if (this.websocketService) {
        this.websocketService.setSimulationMode(this.simulationMode);
      } else {
        console.error('WebSocketService is undefined');
      }
    },
    emitChange(name, value) {
      if (this.isColorControl(name)) {
        value = parseInt(value.replace('#', '0x'), 16);
      }
      this.$emit('control-change', { name, value });
    },
    isColorControl(name) {
      return this.colorControls.some(control => control.name === name);
    },
    resetControls() {
      // Reset all controls to their default values from settings
      window.dispatchEvent(new CustomEvent('resetVisualizationSettings'));
    },
    sendMessage() {
      if (this.chatInput.trim() && this.websocketService) {
        // Store user's message
        this.chatMessages.push({ sender: 'You', message: this.chatInput });
        
        // Send message with TTS flag
        this.websocketService.sendChatMessage({
          message: this.chatInput,
          useOpenAI: true,
          useTTS: this.useOpenAITTS
        });
        
        // Clear input
        this.chatInput = '';
      }
    },
    receiveMessage(message) {
      // Only display message if not using OpenAI TTS
      if (!this.useOpenAITTS) {
        this.chatMessages.push({ sender: 'AI', message });
      }
    },
    toggleFullscreen() {
      this.$emit('toggle-fullscreen');
    },
    enableSpacemouse() {
      this.$emit('enable-spacemouse');
    },
    initializeControls(settings) {
      // Color controls
      this.colorControls = [
        { name: 'nodeColor', label: 'Node Color', type: 'color', value: settings.visualization.nodeColor },
        { name: 'edgeColor', label: 'Edge Color', type: 'color', value: settings.visualization.edgeColor },
        { name: 'hologramColor', label: 'Hologram Color', type: 'color', value: settings.visualization.hologramColor }
      ];

      // Size and opacity controls
      this.sizeOpacityControls = [
        { name: 'nodeSizeScalingFactor', label: 'Node Size Scaling', type: 'range', value: settings.visualization.nodeSizeScalingFactor, min: 1, max: 10, step: 0.1 },
        { name: 'hologramScale', label: 'Hologram Scale', type: 'range', value: settings.visualization.hologramScale, min: 1, max: 10, step: 0.1 },
        { name: 'hologramOpacity', label: 'Hologram Opacity', type: 'range', value: settings.visualization.hologramOpacity, min: 0, max: 1, step: 0.01 },
        { name: 'edgeOpacity', label: 'Edge Opacity', type: 'range', value: settings.visualization.edgeOpacity, min: 0, max: 1, step: 0.01 }
      ];

      // Bloom controls
      this.bloomControls = [
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

      // Force-directed controls
      this.forceDirectedControls = [
        { name: 'forceDirectedIterations', label: 'Iterations', type: 'range', value: settings.visualization.forceDirectedIterations, min: 10, max: 500, step: 10 },
        { name: 'forceDirectedRepulsion', label: 'Repulsion', type: 'range', value: settings.visualization.forceDirectedRepulsion, min: 0.1, max: 10.0, step: 0.1 },
        { name: 'forceDirectedAttraction', label: 'Attraction', type: 'range', value: settings.visualization.forceDirectedAttraction, min: 0.001, max: 0.1, step: 0.001 }
      ];

      // Additional controls
      this.additionalControls = [
        { name: 'labelFontSize', label: 'Label Font Size', type: 'range', value: settings.visualization.labelFontSize, min: 12, max: 72, step: 1 },
        { name: 'fogDensity', label: 'Fog Density', type: 'range', value: settings.visualization.fogDensity, min: 0, max: 0.01, step: 0.0001 }
      ];
    }
  },
  mounted() {
    if (this.websocketService) {
      this.websocketService.on('message', this.receiveMessage);
      this.websocketService.on('simulationModeSet', (mode) => {
        console.log('Simulation mode set to:', mode);
        this.simulationMode = mode;
      });

      // Listen for visualization settings updates
      window.addEventListener('serverSettings', (event) => {
        this.initializeControls(event.detail);
      });
    } else {
      console.error('WebSocketService is undefined');
    }
  },
  beforeUnmount() {
    if (this.websocketService) {
      this.websocketService.off('message', this.receiveMessage);
    }
    window.removeEventListener('serverSettings', this.initializeControls);
  },
  setup() {
    const chatMessagesRef = ref(null);

    const scrollToBottom = () => {
      if (chatMessagesRef.value) {
        chatMessagesRef.value.scrollTop = chatMessagesRef.value.scrollHeight;
      }
    };

    onUpdated(() => {
      scrollToBottom();
    });

    return {
      chatMessagesRef
    };
  }
};
</script>

<style scoped>
#control-panel {
  position: fixed;
  top: 20px;
  right: 0;
  width: 300px;
  max-height: 80vh;
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
}

.panel-content {
  padding: 20px;
}

.control-group {
  margin-bottom: 20px;
}

.control-group h3 {
  margin-bottom: 10px;
  border-bottom: 1px solid #444;
  padding-bottom: 5px;
}

.control-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-bottom: 10px;
}

label {
  font-weight: bold;
}

input[type="color"] {
  width: 100%;
  height: 30px;
  border: none;
  border-radius: 5px;
}

input[type="range"] {
  width: 100%;
}

.range-value {
  font-size: 0.8em;
  text-align: right;
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

select:focus {
  outline: none;
  border-color: #666;
}

option {
  background-color: #333;
  color: white;
}

.reset-button, .control-button {
  width: 100%;
  padding: 10px;
  margin-top: 15px;
  background-color: #333;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.reset-button:hover, .control-button:hover {
  background-color: #555;
}

.chat-interface {
  background-color: #222;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 15px;
}

.chat-messages {
  max-height: 200px;
  overflow-y: auto;
  background-color: #333;
  padding: 10px;
  border-radius: 5px;
}

.chat-message {
  margin-bottom: 10px;
}

.chat-message.user {
  text-align: right;
}

.chat-input-container {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 10px;
}

.chat-input-container input[type="text"] {
  flex-grow: 1;
  padding: 5px;
  border-radius: 3px;
  border: 1px solid #444;
  background-color: #333;
  color: white;
}

.chat-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tts-toggle {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.9em;
  white-space: nowrap;
}

.tts-toggle input[type="checkbox"] {
  margin: 0;
}

.chat-controls button {
  padding: 5px 15px;
  background-color: #444;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
}

.chat-controls button:hover {
  background-color: #555;
}

.button-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

#control-panel::-webkit-scrollbar {
  width: 10px;
}

#control-panel::-webkit-scrollbar-track {
  background: #1e1e1e;
}

#control-panel::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 5px;
}

#control-panel::-webkit-scrollbar-thumb:hover {
  background: #555;
}
</style>
