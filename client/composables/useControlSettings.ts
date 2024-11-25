import { computed } from 'vue';
import type { 
  VisualizationConfig, 
  BloomConfig, 
  FisheyeConfig,
  ControlGroup,
  ControlItem 
} from '@/types/components';
import { useControlGroups } from './useControlGroups';
import { useSettingsStore } from '@/stores/settings';
import { storeToRefs } from 'pinia';

export function useControlSettings() {
  const settingsStore = useSettingsStore();
  const { 
    createControlGroup, 
    createRangeControl, 
    createColorControl, 
    createCheckboxControl 
  } = useControlGroups();

  // Create appearance controls
  const createAppearanceGroup = (config: VisualizationConfig): ControlGroup => {
    const controls: ControlItem[] = [
      createColorControl('node_color', 'Base Node Color', config.node_color),
      createColorControl('node_color_new', 'New Nodes', config.node_color_new),
      createColorControl('node_color_recent', 'Recent Nodes', config.node_color_recent),
      createColorControl('node_color_medium', 'Medium Age', config.node_color_medium),
      createColorControl('node_color_old', 'Old Nodes', config.node_color_old),
      createColorControl('node_color_core', 'Core Nodes', config.node_color_core),
      createColorControl('node_color_secondary', 'Secondary Nodes', config.node_color_secondary),
      createRangeControl('min_node_size', 'Minimum Size', config.min_node_size, 0.05, 0.5, 0.05),
      createRangeControl('max_node_size', 'Maximum Size', config.max_node_size, 0.1, 1.0, 0.1)
    ];

    return createControlGroup('appearance', 'Node Appearance', controls);
  };

  // Create material controls
  const createMaterialGroup = (config: VisualizationConfig): ControlGroup => {
    const controls: ControlItem[] = [
      createRangeControl('node_material_metalness', 'Metalness', config.node_material_metalness, 0, 1, 0.1),
      createRangeControl('node_material_roughness', 'Roughness', config.node_material_roughness, 0, 1, 0.1),
      createRangeControl('node_material_clearcoat', 'Clearcoat', config.node_material_clearcoat, 0, 1, 0.1),
      createRangeControl('node_material_clearcoat_roughness', 'Clearcoat Roughness', config.node_material_clearcoat_roughness, 0, 1, 0.1),
      createRangeControl('node_material_opacity', 'Opacity', config.node_material_opacity, 0, 1, 0.1),
      createRangeControl('node_emissive_min_intensity', 'Min Emissive', config.node_emissive_min_intensity, 0, 1, 0.1),
      createRangeControl('node_emissive_max_intensity', 'Max Emissive', config.node_emissive_max_intensity, 0, 2, 0.1)
    ];

    return createControlGroup('material', 'Material Properties', controls);
  };

  // Create physics controls
  const createPhysicsGroup = (config: VisualizationConfig): ControlGroup => {
    const controls: ControlItem[] = [
      createRangeControl('force_directed_iterations', 'Iterations', config.force_directed_iterations, 100, 500, 10),
      createRangeControl('force_directed_spring', 'Spring Strength', config.force_directed_spring, 0.001, 0.1, 0.001),
      createRangeControl('force_directed_repulsion', 'Repulsion', config.force_directed_repulsion, 100, 2000, 100),
      createRangeControl('force_directed_attraction', 'Attraction', config.force_directed_attraction, 0.001, 0.1, 0.001),
      createRangeControl('force_directed_damping', 'Damping', config.force_directed_damping, 0.1, 1.0, 0.1)
    ];

    return createControlGroup('physics', 'Physics Simulation', controls);
  };

  // Create bloom controls
  const createBloomGroup = (config: BloomConfig): ControlGroup => {
    const controls: ControlItem[] = [
      createRangeControl('node_bloom_strength', 'Node Strength', config.node_bloom_strength, 0, 2, 0.1),
      createRangeControl('node_bloom_radius', 'Node Radius', config.node_bloom_radius, 0, 1, 0.1),
      createRangeControl('node_bloom_threshold', 'Node Threshold', config.node_bloom_threshold, 0, 1, 0.1),
      createRangeControl('edge_bloom_strength', 'Edge Strength', config.edge_bloom_strength, 0, 2, 0.1),
      createRangeControl('edge_bloom_radius', 'Edge Radius', config.edge_bloom_radius, 0, 1, 0.1),
      createRangeControl('edge_bloom_threshold', 'Edge Threshold', config.edge_bloom_threshold, 0, 1, 0.1),
      createRangeControl('environment_bloom_strength', 'Environment Strength', config.environment_bloom_strength, 0, 2, 0.1),
      createRangeControl('environment_bloom_radius', 'Environment Radius', config.environment_bloom_radius, 0, 1, 0.1),
      createRangeControl('environment_bloom_threshold', 'Environment Threshold', config.environment_bloom_threshold, 0, 1, 0.1)
    ];

    return createControlGroup('bloom', 'Bloom Effects', controls);
  };

  // Create environment controls
  const createEnvironmentGroup = (config: VisualizationConfig): ControlGroup => {
    const controls: ControlItem[] = [
      createColorControl('hologram_color', 'Hologram Color', config.hologram_color),
      createRangeControl('hologram_scale', 'Hologram Scale', config.hologram_scale, 1, 10, 1),
      createRangeControl('hologram_opacity', 'Hologram Opacity', config.hologram_opacity, 0, 1, 0.05),
      createRangeControl('fog_density', 'Fog Density', config.fog_density, 0, 0.01, 0.0001)
    ];

    return createControlGroup('environment', 'Environment', controls);
  };

  // Create fisheye controls
  const createFisheyeGroup = (config: FisheyeConfig): ControlGroup => {
    const controls: ControlItem[] = [
      createCheckboxControl('enabled', 'Enable Fisheye', config.enabled),
      createRangeControl('strength', 'Strength', config.strength, 0, 1, 0.1),
      createRangeControl('radius', 'Radius', config.radius, 10, 200, 10),
      createRangeControl('focus_x', 'Focus X', config.focus_x, -100, 100, 1),
      createRangeControl('focus_y', 'Focus Y', config.focus_y, -100, 100, 1),
      createRangeControl('focus_z', 'Focus Z', config.focus_z, -100, 100, 1)
    ];

    return createControlGroup('fisheye', 'Fisheye Effect', controls);
  };

  // Handle control changes
  const handleControlChange = (groupName: string, controlName: string, value: any) => {
    switch (groupName) {
      case 'appearance':
      case 'material':
      case 'physics':
      case 'environment':
        settingsStore.updateVisualizationSettings({ [controlName]: value });
        break;
      case 'bloom':
        settingsStore.updateBloomSettings({ [controlName]: value });
        break;
      case 'fisheye':
        settingsStore.updateFisheyeSettings({ [controlName]: value });
        break;
    }
  };

  return {
    createAppearanceGroup,
    createMaterialGroup,
    createPhysicsGroup,
    createBloomGroup,
    createEnvironmentGroup,
    createFisheyeGroup,
    handleControlChange
  };
}
