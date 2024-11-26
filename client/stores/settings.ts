import { defineStore } from 'pinia';
import type { 
  VisualizationConfig, 
  BloomConfig, 
  FisheyeConfig 
} from '../types/components';
import { 
  DEFAULT_VISUALIZATION_CONFIG,
  DEFAULT_BLOOM_CONFIG,
  DEFAULT_FISHEYE_CONFIG
} from '../types/components';

interface SettingsState {
  visualization: VisualizationConfig;
  bloom: BloomConfig;
  fisheye: FisheyeConfig;
  isDirty: boolean;
}

export const useSettingsStore = defineStore('settings', {
  state: (): SettingsState => ({
    visualization: { ...DEFAULT_VISUALIZATION_CONFIG },
    bloom: { ...DEFAULT_BLOOM_CONFIG },
    fisheye: { ...DEFAULT_FISHEYE_CONFIG },
    isDirty: false
  }),

  getters: {
    getVisualizationSettings: (state) => state.visualization,
    getBloomSettings: (state) => state.bloom,
    getFisheyeSettings: (state) => state.fisheye,
    hasUnsavedChanges: (state) => state.isDirty
  },

  actions: {
    updateVisualizationSettings(settings: Partial<VisualizationConfig>) {
      this.visualization = {
        ...this.visualization,
        ...settings
      };
      this.isDirty = true;
    },

    updateBloomSettings(settings: Partial<BloomConfig>) {
      this.bloom = {
        ...this.bloom,
        ...settings
      };
      this.isDirty = true;
    },

    updateFisheyeSettings(settings: Partial<FisheyeConfig>) {
      this.fisheye = {
        ...this.fisheye,
        ...settings
      };
      this.isDirty = true;
    },

    applyServerSettings(settings: {
      visualization?: Partial<VisualizationConfig>;
      bloom?: Partial<BloomConfig>;
      fisheye?: Partial<FisheyeConfig>;
    }) {
      if (settings.visualization) {
        this.visualization = {
          ...this.visualization,
          ...settings.visualization
        };
      }
      if (settings.bloom) {
        this.bloom = {
          ...this.bloom,
          ...settings.bloom
        };
      }
      if (settings.fisheye) {
        this.fisheye = {
          ...this.fisheye,
          ...settings.fisheye
        };
      }
      this.isDirty = false;
    },

    resetToDefaults() {
      this.visualization = { ...DEFAULT_VISUALIZATION_CONFIG };
      this.bloom = { ...DEFAULT_BLOOM_CONFIG };
      this.fisheye = { ...DEFAULT_FISHEYE_CONFIG };
      this.isDirty = true;
    },

    markSaved() {
      this.isDirty = false;
    }
  }
});
