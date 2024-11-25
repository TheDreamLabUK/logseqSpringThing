<template>
  <div 
    v-if="selectedNode"
    class="node-info-panel"
  >
    <h3>Node Information</h3>
    <p><strong>ID:</strong> {{ selectedNode.id }}</p>
    <p><strong>Name:</strong> {{ selectedNode.name }}</p>
    <template v-if="selectedNode.metadata">
      <div v-for="(value, key) in selectedNode.metadata" :key="key" class="metadata-item">
        <strong>{{ formatKey(key) }}:</strong> {{ formatValue(value) }}
      </div>
    </template>
  </div>
</template>

<script lang="ts">
import { defineComponent, computed } from 'vue';
import { useVisualizationStore } from '../stores/visualization';

export default defineComponent({
  name: 'NodeInfoPanel',
  
  setup() {
    const store = useVisualizationStore();
    const selectedNode = computed(() => store.selectedNode);

    const formatKey = (key: string) => {
      return key
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    };

    const formatValue = (value: any) => {
      if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
      }
      return value;
    };

    return {
      selectedNode,
      formatKey,
      formatValue
    };
  }
});
</script>

<style scoped>
.node-info-panel {
  position: absolute;
  top: 20px;
  left: 20px;
  width: 300px;
  max-height: 40vh;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  overflow-y: auto;
}

.metadata-item {
  margin: 5px 0;
}

h3 {
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 1.2em;
}

p {
  margin: 8px 0;
}

strong {
  font-weight: 600;
}
</style>
