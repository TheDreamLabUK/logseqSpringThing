<template>
  <div id="control-panel" :class="{ hidden: isHidden }">
    <button @click="togglePanel">Toggle Panel</button>
    <div v-for="control in controls" :key="control.name">
      <label :for="control.name">{{ control.label }}</label>
      <input
        :id="control.name"
        :type="control.type"
        v-model="control.value"
        @change="emitChange(control.name, control.value)"
      />
    </div>
    <button @click="resetControls">Reset</button>
  </div>
</template>

<script>
export default {
  props: {
    // Define any props if necessary
  },
  data() {
    return {
      isHidden: false,
      controls: [
        { name: 'node_color', label: 'Node Color', type: 'color', value: '#1A0B31' },
        { name: 'edge_color', label: 'Edge Color', type: 'color', value: '#ff0000' },
        // Add more controls as needed
      ],
    };
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
      const defaults = {
        node_color: '#1A0B31',
        edge_color: '#ff0000',
        // Define default values for other controls
      };
      return defaults[name] || '';
    },
  },
};
</script>

<style scoped>
#control-panel {
  /* Your existing styles */
}
.hidden {
  display: none;
}
</style>