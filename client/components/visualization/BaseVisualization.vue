<template>
  <renderer
    :antialias="true"
    :xr="platformInfo.hasXRSupport"
    :size="{ w: windowSize.width, h: windowSize.height }"
    ref="renderer"
  >
    <scene ref="scene">
      <!-- Camera System -->
      <camera
        :position="cameraPosition"
        :fov="75"
        :aspect="aspect"
        :near="0.1"
        :far="1000"
        ref="camera"
      />

      <!-- Lighting -->
      <ambient-light :intensity="1.5" />
      <directional-light
        :position="{ x: 10, y: 20, z: 10 }"
        :intensity="2.0"
        :cast-shadow="true"
      />
      <hemisphere-light
        :sky-color="0xffffff"
        :ground-color="0x444444"
        :intensity="1.5"
      />

      <!-- Graph Visualization -->
      <graph-system
        v-if="graphData"
        :nodes="graphData.nodes"
        :edges="graphData.edges"
        :settings="visualSettings"
      />

      <!-- XR Controllers and Hands -->
      <template v-if="platformInfo.isQuest">
        <xr-controllers />
        <xr-hands />
      </template>

      <!-- Effects -->
      <effects-system :settings="effectsSettings">
        <bloom v-if="effectsSettings.bloom.enabled" v-bind="effectsSettings.bloom" />
        <ssao v-if="effectsSettings.ssao.enabled" v-bind="effectsSettings.ssao" />
      </effects-system>
    </scene>
  </renderer>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { usePlatform } from '@/composables/usePlatform';
import type { GraphData } from '@/types/core';

export default defineComponent({
  name: 'BaseVisualization',

  setup() {
    // Refs for Three.js components
    const renderer = ref(null);
    const scene = ref(null);
    const camera = ref(null);

    // Platform and settings
    const { getPlatformInfo } = usePlatform();
    const settingsStore = useSettingsStore();
    const platformInfo = computed(() => getPlatformInfo());

    // Window size reactive state
    const windowSize = ref({
      width: window.innerWidth,
      height: window.innerHeight
    });
    const aspect = computed(() => windowSize.value.width / windowSize.value.height);

    // Camera position with reactive updates
    const cameraPosition = ref({ x: 0, y: 75, z: 200 });

    // Graph data
    const graphData = ref<GraphData | null>(null);

    // Settings from store
    const visualSettings = computed(() => settingsStore.getVisualizationSettings);
    const effectsSettings = computed(() => ({
      bloom: settingsStore.getBloomSettings,
      ssao: {
        enabled: false, // Add SSAO settings to store if needed
        radius: 0.5,
        intensity: 1.0,
        bias: 0.5
      }
    }));

    // Window resize handler
    const handleResize = () => {
      windowSize.value = {
        width: window.innerWidth,
        height: window.innerHeight
      };
    };

    // Lifecycle hooks
    onMounted(() => {
      window.addEventListener('resize', handleResize);
      
      // Initialize platform-specific features
      if (platformInfo.value.isQuest) {
        initializeXR();
      }
    });

    onBeforeUnmount(() => {
      window.removeEventListener('resize', handleResize);
    });

    // XR initialization
    const initializeXR = async () => {
      if (!renderer.value || !platformInfo.value.hasXRSupport) return;

      try {
        const session = await navigator.xr?.requestSession('immersive-vr', {
          requiredFeatures: ['local-floor', 'bounded-floor'],
          optionalFeatures: ['hand-tracking']
        });

        if (session) {
          await renderer.value.xr.setSession(session);
        }
      } catch (error) {
        console.error('Failed to initialize XR:', error);
      }
    };

    // Public methods
    const updateGraphData = (data: GraphData) => {
      graphData.value = data;
    };

    return {
      // Template refs
      renderer,
      scene,
      camera,

      // Computed properties
      platformInfo,
      windowSize,
      aspect,
      cameraPosition,
      graphData,
      visualSettings,
      effectsSettings,

      // Methods
      updateGraphData
    };
  }
});
</script>

<style scoped>
.renderer-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
</style>
