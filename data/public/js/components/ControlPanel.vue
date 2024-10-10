s<template>
  <div id="control-panel" :class="{ hidden: isHidden }">
    <button @click="hidePanel" class="hide-button">Hide Panel</button>
    <div class="controls-container">
      <div v-for="control in controls" :key="control.name" class="control-item">
        <label :for="control.name">{{ control.label }}</label>
        <input v-if="control.type === 'color'"
               :id="control.name"
               type="color"
               v-model="control.value"
               @change="emitChange(control.name, control.value)"
        >
        <input v-else-if="control.type === 'range'"
               :id="control.name"
               type="range"
               v-model.number="control.value"
               :min="control.min"
               :max="control.max"
               :step="control.step"
               @input="emitChange(control.name, control.value)"
        >
        <span v-if="control.type === 'range'" class="range-value">{{ control.value }}</span>
      </div>
    </div>
    <button @click="resetControls" class="reset-button">Reset to Defaults</button>
  </div>
</template>

<script>
export default {
  name: 'ControlPanel',
  data() {
    return {
      isHidden: false,
      controls: [
        { name: 'node_color', label: 'Node Color', type: 'color', value: '#1A0B31' },
        { name: 'edge_color', label: 'Edge Color', type: 'color', value: '#ff0000' },
        { name: 'hologram_color', label: 'Hologram Color', type: 'color', value: '#FFD700' },
        { name: 'node_size_scaling_factor', label: 'Node Size Scaling', type: 'range', value: 1000, min: 100, max: 2000, step: 10 },
        { name: 'hologram_scale', label: 'Hologram Scale', type: 'range', value: 5, min: 1, max: 10, step: 0.1 },
        { name: 'hologram_opacity', label: 'Hologram Opacity', type: 'range', value: 0.1, min: 0, max: 1, step: 0.01 },
        { name: 'edge_opacity', label: 'Edge Opacity', type: 'range', value: 0.3, min: 0, max: 1, step: 0.01 },
        { name: 'label_font_size', label: 'Label Font Size', type: 'range', value: 36, min: 12, max: 72, step: 1 },
        { name: 'fog_density', label: 'Fog Density', type: 'range', value: 0.002, min: 0, max: 0.01, step: 0.0001 },
        { name: 'node_bloom_strength', label: 'Node Bloom Strength', type: 'range', value: 0.1, min: 0, max: 1, step: 0.01 },
        { name: 'node_bloom_radius', label: 'Node Bloom Radius', type: 'range', value: 0.1, min: 0, max: 1, step: 0.01 },
        { name: 'node_bloom_threshold', label: 'Node Bloom Threshold', type: 'range', value: 0, min: 0, max: 1, step: 0.01 },
        { name: 'edge_bloom_strength', label: 'Edge Bloom Strength', type: 'range', value: 0.2, min: 0, max: 1, step: 0.01 },
        { name: 'edge_bloom_radius', label: 'Edge Bloom Radius', type: 'range', value: 0.3, min: 0, max: 1, step: 0.01 },
        { name: 'edge_bloom_threshold', label: 'Edge Bloom Threshold', type: 'range', value: 0, min: 0, max: 1, step: 0.01 },
        { name: 'environment_bloom_strength', label: 'Environment Bloom Strength', type: 'range', value: 1, min: 0, max: 2, step: 0.01 },
        { name: 'environment_bloom_radius', label: 'Environment Bloom Radius', type: 'range', value: 1, min: 0, max: 2, step: 0.01 },
        { name: 'environment_bloom_threshold', label: 'Environment Bloom Threshold', type: 'range', value: 0, min: 0, max: 1, step: 0.01 },
      ],
    };
  },
  methods: {
    hidePanel() {
      this.isHidden = true;
    },
    emitChange(name, value) {
      // Convert color hex to 0xRRGGBB format for Three.js
      if (this.isColorControl(name)) {
        value = parseInt(value.replace('#', '0x'), 16);
      }
      this.$emit('control-change', { name, value });
    },
    isColorControl(name) {
      return ['node_color', 'edge_color', 'hologram_color'].includes(name);
    },
    resetControls() {
      this.controls.forEach(control => {
        control.value = this.getDefaultValue(control.name);
        this.emitChange(control.name, control.value);
      });
    },
    getDefaultValue(name) {
      // Default values from settings.toml
      const defaults = {
        node_color: '#1A0B31',
        edge_color: '#ff0000',
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
      };
      return defaults[name] || '';
    },
  },
};
</script>

<style scoped>
#control-panel {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 300px;
  max-height: 80vh;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 20px;
  border-radius: 10px;
  overflow-y: auto;
  z-index: 1000;
  transition: transform 0.3s ease-in-out;
}

#control-panel.hidden {
  transform: translateX(320px);
}

.controls-container {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.control-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
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

.hide-button, .reset-button {
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

.hide-button:hover, .reset-button:hover {
  background-color: #555;
}

/* Scrollbar styling */
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