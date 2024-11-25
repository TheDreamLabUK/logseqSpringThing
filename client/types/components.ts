// Control Panel Types
export interface ControlGroup {
  label: string;
  name: string;
  controls: ControlItem[];
  collapsed: boolean;
}

export interface ControlItem {
  name: string;
  label: string;
  value: number | string | boolean;
  type: 'range' | 'color' | 'checkbox' | 'select';
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}

// Visualization Config Types
export interface VisualizationConfig {
  // Node appearance
  node_color: string;
  node_color_new: string;
  node_color_recent: string;
  node_color_medium: string;
  node_color_old: string;
  node_color_core: string;
  node_color_secondary: string;
  min_node_size: number;
  max_node_size: number;

  // Edge appearance
  edge_color: string;
  edge_opacity: number;
  edge_min_width: number;
  edge_max_width: number;

  // Material properties
  material: {
    node_material_metalness: number;
    node_material_roughness: number;
    node_material_clearcoat: number;
    node_material_clearcoat_roughness: number;
    node_material_opacity: number;
    node_emissive_min_intensity: number;
    node_emissive_max_intensity: number;
  };

  // Physics simulation
  physics: {
    force_directed_iterations: number;
    force_directed_spring: number;
    force_directed_repulsion: number;
    force_directed_attraction: number;
    force_directed_damping: number;
  };

  // Label settings
  label_font_size: number;
  label_font_family: string;
  label_padding: number;
  label_vertical_offset: number;
  label_close_offset: number;
  label_background_color: string;
  label_text_color: string;
  label_info_text_color: string;
  label_xr_font_size: number;

  // Environment settings
  fog_density: number;
  hologram_color: string;
  hologram_scale: number;
  hologram_opacity: number;
}

// Effect Settings
export interface BloomConfig {
  enabled: boolean;
  strength: number;
  radius: number;
  threshold: number;
  node_bloom_strength: number;
  node_bloom_radius: number;
  node_bloom_threshold: number;
  edge_bloom_strength: number;
  edge_bloom_radius: number;
  edge_bloom_threshold: number;
  environment_bloom_strength: number;
  environment_bloom_radius: number;
  environment_bloom_threshold: number;
}

export interface FisheyeConfig {
  enabled: boolean;
  strength: number;
  radius: number;
  focus_x: number;
  focus_y: number;
  focus_z: number;
}

export interface ControlPanelProps {
  visualizationConfig: VisualizationConfig;
  bloomConfig: BloomConfig;
  fisheyeConfig: FisheyeConfig;
}

export interface ControlPanelEmits {
  (event: 'update:visualizationConfig', value: Partial<VisualizationConfig>): void;
  (event: 'update:bloomConfig', value: Partial<BloomConfig>): void;
  (event: 'update:fisheyeConfig', value: Partial<FisheyeConfig>): void;
  (event: 'saveSettings'): void;
}

// Default Values
export const DEFAULT_VISUALIZATION_CONFIG: VisualizationConfig = {
  // Node appearance
  node_color: '#FFA500',
  node_color_new: '#FFD700',
  node_color_recent: '#FFA500',
  node_color_medium: '#DAA520',
  node_color_old: '#CD853F',
  node_color_core: '#FFB90F',
  node_color_secondary: '#FFC125',
  min_node_size: 0.15,
  max_node_size: 0.4,

  // Edge appearance
  edge_color: '#FFD700',
  edge_opacity: 0.4,
  edge_min_width: 1.5,
  edge_max_width: 6.0,

  // Material properties
  material: {
    node_material_metalness: 0.7,
    node_material_roughness: 0.2,
    node_material_clearcoat: 0.8,
    node_material_clearcoat_roughness: 0.1,
    node_material_opacity: 0.95,
    node_emissive_min_intensity: 0.4,
    node_emissive_max_intensity: 1.0
  },

  // Physics simulation
  physics: {
    force_directed_iterations: 300,
    force_directed_spring: 0.015,
    force_directed_repulsion: 1200.0,
    force_directed_attraction: 0.012,
    force_directed_damping: 0.85
  },

  // Label settings
  label_font_size: 42,
  label_font_family: 'Arial',
  label_padding: 24,
  label_vertical_offset: 2.5,
  label_close_offset: 0.25,
  label_background_color: 'rgba(0, 0, 0, 0.85)',
  label_text_color: 'white',
  label_info_text_color: 'lightgray',
  label_xr_font_size: 28,

  // Environment settings
  fog_density: 0.001,
  hologram_color: '#FFC125',
  hologram_scale: 6.0,
  hologram_opacity: 0.15
};

export const DEFAULT_BLOOM_CONFIG: BloomConfig = {
  enabled: true,
  strength: 1.5,
  radius: 0.4,
  threshold: 0.2,
  node_bloom_strength: 1.5,
  node_bloom_radius: 0.4,
  node_bloom_threshold: 0.2,
  edge_bloom_strength: 1.2,
  edge_bloom_radius: 0.4,
  edge_bloom_threshold: 0.2,
  environment_bloom_strength: 1.0,
  environment_bloom_radius: 0.4,
  environment_bloom_threshold: 0.2
};

export const DEFAULT_FISHEYE_CONFIG: FisheyeConfig = {
  enabled: false,
  strength: 0.5,
  radius: 100.0,
  focus_x: 0.0,
  focus_y: 0.0,
  focus_z: 0.0
};
