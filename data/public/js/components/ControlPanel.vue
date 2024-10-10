<template>
  <div :class="['control-panel', { hidden: isHidden, 'vr-mode': isVR }]">
    <button @click="togglePanel" class="toggle-button">{{ isHidden ? 'Show' : 'Hide' }} Panel</button>
    <div v-if="!isHidden" class="controls-container">
      <div v-for="control in controls" :key="control.name" class="control-group">
        <label :for="control.name">{{ control.label }}</label>
        <input
          v-if="control.type === 'slider'"
          type="range"
          :id="control.name"
          :min="control.min"
          :max="control.max"
          :step="control.step"
          v-model.number="control.value"
          @input="emitChange(control.name, control.value)"
        />
        <input
          v-else-if="control.type === 'color'"
          type="color"
          :id="control.name"
          v-model="control.value"
          @input="emitChange(control.name, control.value)"
        />
      </div>
      <button @click="resetControls" class="reset-button">Reset to Defaults</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ControlPanel',
  props: {
    mode: {
      type: String,
      default: 'desktop', // 'vr' or 'desktop'
    },
  },
  data() {
    return {
      isHidden: false,
      controls: [
        { name: 'node_color', label: 'Node Color', type: 'color', value: '#1A0B31' },
        { name: 'edge_color', label: 'Edge Color', type: 'color', value: '#FF0000' },
        { name: 'hologram_color', label: 'Hologram Color', type: 'color', value: '#FFD700' },
        { name: 'node_size_scaling_factor', label: 'Node Size Scaling Factor', type: 'slider', min: 100, max: 2000, step: 10, value: 1000 },
        { name: 'hologram_scale', label: 'Hologram Scale', type: 'slider', min: 1, max: 10, step: 0.1, value: 5 },
        { name: 'hologram_opacity', label: 'Hologram Opacity', type: 'slider', min: 0, max: 1, step: 0.01, value: 0.1 },
        { name: 'edge_opacity', label: 'Edge Opacity', type: 'slider', min: 0, max: 1, step: 0.01, value: 0.3 },
        { name: 'label_font_size', label: 'Label Font Size', type: 'slider', min: 12, max: 72, step: 1, value: 36 },
        { name: 'fog_density', label: 'Fog Density', type: 'slider', min: 0, max: 0.01, step: 0.0001, value: 0.002 },
        { name: 'node_bloom_strength', label: 'Node Bloom Strength', type: 'slider', min: 0, max: 3, step: 0.1, value: 0.1 },
        { name: 'node_bloom_radius', label: 'Node Bloom Radius', type: 'slider', min: 0, max: 3, step: 0.1, value: 0.1 },
        { name: 'node_bloom_threshold', label: 'Node Bloom Threshold', type: 'slider', min: 0, max: 1, step: 0.01, value: 0 },
        { name: 'edge_bloom_strength', label: 'Edge Bloom Strength', type: 'slider', min: 0, max: 3, step: 0.1, value: 0.2 },
        { name: 'edge_bloom_radius', label: 'Edge Bloom Radius', type: 'slider', min: 0, max: 3, step: 0.1, value: 0.3 },
        { name: 'edge_bloom_threshold', label: 'Edge Bloom Threshold', type: 'slider', min: 0, max: 1, step: 0.01, value: 0 },
        { name: 'environment_bloom_strength', label: 'Environment Bloom Strength', type: 'slider', min: 0, max: 3, step: 0.1, value: 1 },
        { name: 'environment_bloom_radius', label: 'Environment Bloom Radius', type: 'slider', min: 0, max: 3, step: 0.1, value: 1 },
        { name: 'environment_bloom_threshold', label: 'Environment Bloom Threshold', type: 'slider', min: 0, max: 1, step: 0.01, value: 0 },
      ],
      defaultValues: {
        node_color: '#1A0B31',
        edge_color: '#FF0000',
        hologram_color: '#FFD700',
        node_size_scaling_factor: 1000,
        hologram_scale: 5,
        hologram_opacity: 0.1,
        edge_opacity: 0.3,
        label_font_size: 36,
        fog_density: 0.002,
        node_bloom_strength: 0.1,
        node_bloom_radius: 0.1,
        node_bloom_threshold: 0,
        edge_bloom_strength: 0.2,
        edge_bloom_radius: 0.3,
        edge_bloom_threshold: 0,
        environment_bloom_strength: 1,
        environment_bloom_radius: 1,
        environment_bloom_threshold: 0,
      },
    };
  },
  computed: {
    isVR() {
      return this.mode === 'vr';
    },
  },
  watch: {
    mode(newMode) {
      this.adjustForMode(newMode);
    },
  },
  methods: {
    togglePanel() {
      this.isHidden = !this.isHidden;
    },
    emitChange(name, value) {
      this.$emit('control-change', { name, value });
    },
    resetControls() {
      this.controls.forEach(control => {
        control.value = this.getDefaultValue(control.name);
        this.emitChange(control.name, control.value);
      });
    },
    getDefaultValue(name) {
      return this.defaultValues[name] || null;
    },
    adjustForMode(mode) {
      if (mode === 'vr') {
        // Apply VR-specific adjustments
        this.isHidden = false; // Always show in VR mode
        // Additional VR-specific adjustments can be added here
      } else {
        // Revert to desktop settings if necessary
      }
    },
  },
  mounted() {
    this.adjustForMode(this.mode);
  },
};
</script>

<style scoped>
.control-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 300px;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.9);
  box-shadow: -2px 0 5px rgba(0, 0, 0, 0.3);
  transition: transform 0.3s ease-in-out;
  z-index: 1000;
}

.control-panel.hidden {
  transform: translateX(100%);
}

.control-panel.vr-mode {
  position: relative;
  width: auto;
  height: auto;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 20px;
  border-radius: 10px;
}

.toggle-button {
  position: absolute;
  left: -60px;
  top: 10px;
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 10px;
  cursor: pointer;
}

.controls-container {
  padding: 20px;
  overflow-y: auto;
  height: calc(100% - 40px);
}

.control-group {
  margin-bottom: 15px;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

input[type="range"] {
  width: 100%;
}

input[type="color"] {
  width: 100%;
  height: 30px;
  border: none;
  padding: 0;
  background: none;
}

.reset-button {
  display: block;
  width: 100%;
  padding: 10px;
  background-color: #f44336;
  color: white;
  border: none;
  cursor: pointer;
  margin-top: 20px;
}

/* VR mode specific styles */
.vr-mode .control-group {
  margin-bottom: 25px;
}

.vr-mode label {
  font-size: 1.2em;
}

.vr-mode input[type="range"] {
  height: 30px;
}

.vr-mode .reset-button {
  font-size: 1.2em;
  padding: 15px;
}
</style>
