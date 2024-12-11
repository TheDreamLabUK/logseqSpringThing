<template>
  <div id="control-panel" :class="{ hidden: isHidden }">
    <button class="toggle-button" @click="togglePanel">
      {{ isHidden ? 'Show Controls' : 'Hide Controls' }}
    </button>
    
    <div class="panel-content">
      <!-- Node Appearance -->
      <div v-for="group in controlGroups" :key="group.name" class="control-group">
        <div class="group-header" @click="toggleGroup(group.name)">
          <h3>{{ group.label }}</h3>
        </div>
        <div v-if="!collapsedGroups[group.name]" class="group-content">
          <div v-for="control in group.controls" :key="control.name" class="control-item">
            <label>{{ control.label }}</label>
            
            <!-- Color Input -->
            <template v-if="control.type === 'color'">
              <input type="color"
                     :value="control.value"
                     @input="($event: Event) => handleColorInput($event, group.name, control.name)">
            </template>
            
            <!-- Range Input -->
            <template v-if="control.type === 'range'">
              <input type="range"
                     :min="control.min"
                     :max="control.max"
                     :step="control.step"
                     :value="control.value"
                     @input="($event: Event) => handleRangeInput($event, group.name, control.name)">
              <span class="range-value">{{ typeof control.value === 'number' ? control.value.toFixed(2) : control.value }}</span>
            </template>
            
            <!-- Checkbox Input -->
            <template v-if="control.type === 'checkbox'">
              <input type="checkbox"
                     :value="control.value"
                     :checked="Boolean(control.value)"
                     @change="($event: Event) => handleCheckboxChange($event, group.name, control.name)">
            </template>
          </div>
        </div>
      </div>

      <!-- Save Settings Button -->
      <button class="save-button" 
              @click="saveSettings"
              :disabled="!hasUnsavedChanges || isSaving">
        {{ getSaveButtonText() }}
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import { useSettingsStore } from '../stores/settings';
import { useWebSocketStore } from '../stores/websocket';
import { useControlGroups } from '../composables/useControlGroups';
import { useControlSettings } from '../composables/useControlSettings';
import { SERVER_MESSAGE_TYPES, MESSAGE_FIELDS } from '../constants/websocket';
import type { ControlGroup } from '../types/components';
import type { SettingsUpdateMessage, MaterialSettings, BloomSettings, FisheyeSettings } from '../types/websocket';

export default defineComponent({
  name: 'ControlPanel',
  
  setup() {
    const settingsStore = useSettingsStore();
    const websocketStore = useWebSocketStore();
    const { collapsedGroups, toggleGroup } = useControlGroups();
    const { 
      createAppearanceGroup,
      createMaterialGroup,
      createPhysicsGroup,
      createBloomGroup,
      createEnvironmentGroup,
      createFisheyeGroup,
      handleControlChange
    } = useControlSettings();

    const isHidden = ref(false);
    const isSaving = ref(false);
    const { visualization, bloom, fisheye, isDirty: hasUnsavedChanges } = storeToRefs(settingsStore);

    // Map visualization settings to material settings
    const getMaterialSettings = computed((): Partial<MaterialSettings> => ({
      nodeSize: (visualization.value.min_node_size + visualization.value.max_node_size) / 2,
      nodeColor: visualization.value.node_color,
      edgeWidth: (visualization.value.edge_min_width + visualization.value.edge_max_width) / 2,
      edgeColor: visualization.value.edge_color,
      highlightColor: visualization.value.node_color_core,
      opacity: visualization.value.material.node_material_opacity
    }));

    // Compute control groups based on current settings
    const controlGroups = computed<ControlGroup[]>(() => [
      createAppearanceGroup(visualization.value),
      createMaterialGroup(visualization.value),
      createPhysicsGroup(visualization.value),
      createBloomGroup(bloom.value),
      createEnvironmentGroup(visualization.value),
      createFisheyeGroup(fisheye.value)
    ]);

    const togglePanel = () => {
      isHidden.value = !isHidden.value;
    };

    // Type-safe event handlers
    const handleColorInput = (event: Event, groupName: string, controlName: string) => {
      const input = event.target as HTMLInputElement;
      handleControlChange(groupName, controlName, input.value);
    };

    const handleRangeInput = (event: Event, groupName: string, controlName: string) => {
      const input = event.target as HTMLInputElement;
      handleControlChange(groupName, controlName, parseFloat(input.value));
    };

    const handleCheckboxChange = (event: Event, groupName: string, controlName: string) => {
      const input = event.target as HTMLInputElement;
      handleControlChange(groupName, controlName, input.checked);
    };

    const getSaveButtonText = () => {
      if (isSaving.value) return 'Saving...';
      if (!hasUnsavedChanges.value) return 'No Changes';
      return 'Save Settings';
    };

    const saveSettings = async () => {
      if (!hasUnsavedChanges.value || isSaving.value) return;
      
      try {
        isSaving.value = true;
        
        // Prepare settings update message
        const updateMessage: SettingsUpdateMessage = {
          type: SERVER_MESSAGE_TYPES.UPDATE_SETTINGS,
          settings: {
            [MESSAGE_FIELDS.MATERIAL]: getMaterialSettings.value,
            [MESSAGE_FIELDS.BLOOM]: bloom.value,
            [MESSAGE_FIELDS.FISHEYE]: fisheye.value
          }
        };

        // Send settings through WebSocket
        websocketStore.send(updateMessage);
        
        // Mark settings as saved in store
        settingsStore.markSaved();
        
        console.debug('Settings update sent:', updateMessage);
      } catch (error) {
        console.error('Failed to save settings:', error);
      } finally {
        isSaving.value = false;
      }
    };

    onMounted(() => {
      console.debug('ControlPanel mounted, initial settings:', {
        material: getMaterialSettings.value,
        bloom: bloom.value,
        fisheye: fisheye.value
      });
    });

    return {
      isHidden,
      isSaving,
      collapsedGroups,
      controlGroups,
      hasUnsavedChanges,
      togglePanel,
      toggleGroup,
      handleColorInput,
      handleRangeInput,
      handleCheckboxChange,
      saveSettings,
      getSaveButtonText
    };
  }
});
</script>

<style scoped>
/* Styles remain unchanged */
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
  z-index: 1001;
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
  margin-top: 20px;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.2s;
}

.save-button:hover:not(:disabled) {
  background-color: #218838;
}

.save-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
  opacity: 0.65;
}
</style>
