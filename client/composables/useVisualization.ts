import { ref, computed, onBeforeUnmount } from 'vue';
import THREE from '../utils/three';
import { useSettingsStore } from '../stores/settings';
import { useGraphSystem } from './useGraphSystem';
import { useEffectsSystem } from './useEffectsSystem';
import { usePlatform } from './usePlatform';
import { useThreeScene } from './useThreeScene';
import type { Node, Edge, Transform, CoreState, InitializationOptions } from '../types/core';

export function useVisualization() {
  const settingsStore = useSettingsStore();
  const { getPlatformInfo, initializePlatform, getState } = usePlatform();
  const { resources: threeResources, initScene } = useThreeScene();
  
  // Core state
  const state = ref<CoreState>({
    renderer: null,
    camera: null,
    scene: null,
    canvas: null,
    isInitialized: false,
    isXRSupported: false,
    isWebGL2: false
  });

  // Animation frame ID
  let animationFrameId: number | null = null;

  // Systems
  const { 
    nodes, 
    edges,
    updateGraphData,
    getNodePosition,
    getNodeScale,
    getNodeColor 
  } = useGraphSystem();

  const {
    composer,
    render: renderEffects,
    updateBloomSettings
  } = useEffectsSystem(
    state.value.renderer as THREE.WebGLRenderer,
    state.value.scene as THREE.Scene,
    state.value.camera as THREE.PerspectiveCamera
  );

  // Settings
  const settings = computed(() => settingsStore.getVisualizationSettings);
  const platformInfo = computed(() => getPlatformInfo());

  // Animation loop
  const animate = () => {
    if (!state.value.isInitialized || !threeResources.value) return;

    if (threeResources.value.controls) {
      threeResources.value.controls.update();
    }

    // Render effects if available, otherwise render normally
    if (composer.value) {
      renderEffects();
    } else if (threeResources.value.renderer && threeResources.value.scene && threeResources.value.camera) {
      threeResources.value.renderer.render(threeResources.value.scene, threeResources.value.camera);
    }

    animationFrameId = requestAnimationFrame(animate);
  };

  // Platform initialization
  const initialize = async (options: InitializationOptions) => {
    if (state.value.isInitialized) return;

    try {
      console.log('Initializing Three.js scene...');
      // Initialize Three.js scene and wait for it to complete
      const resources = await initScene();
      
      if (!resources) {
        throw new Error('Failed to initialize Three.js scene');
      }

      console.log('Initializing platform...');
      // Initialize platform
      await initializePlatform(options);
      
      // Get platform state
      const platformState = getState();
      if (!platformState) {
        throw new Error('Failed to initialize platform state');
      }

      // Update state with scene resources
      state.value = {
        renderer: resources.renderer,
        camera: resources.camera,
        scene: resources.scene,
        canvas: resources.renderer.domElement,
        isInitialized: true,
        isXRSupported: platformState.isXRSupported,
        isWebGL2: platformState.isWebGL2
      };

      console.log('Updating settings...');
      // Update settings
      updateFromSettings(settings.value);

      console.log('Starting animation loop...');
      // Start animation loop
      animate();

      console.log('Visualization system initialized successfully');

    } catch (error) {
      console.error('Failed to initialize visualization:', error);
      throw error;
    }
  };

  // Update from settings
  const updateFromSettings = (newSettings: any) => {
    if (!state.value.isInitialized) return;

    // Update scene properties
    if (state.value.scene?.fog && 'fog_density' in newSettings) {
      (state.value.scene.fog as THREE.FogExp2).density = newSettings.fog_density;
    }

    // Update effects
    updateBloomSettings();
  };

  // Update graph data
  const updateGraph = (graphData: { nodes: Node[], edges: Edge[] }) => {
    console.log('Updating graph data:', {
      nodes: graphData.nodes.length,
      edges: graphData.edges.length
    });
    updateGraphData(graphData);
  };

  // Cleanup
  const dispose = () => {
    console.log('Disposing visualization system...');
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    if (!state.value.isInitialized) return;

    state.value.scene?.traverse((object: THREE.Object3D) => {
      if (object instanceof THREE.Mesh) {
        object.geometry?.dispose();
        if (object.material instanceof THREE.Material) {
          object.material.dispose();
        }
      }
    });

    state.value.renderer?.dispose();
    state.value = {
      renderer: null,
      camera: null,
      scene: null,
      canvas: null,
      isInitialized: false,
      isXRSupported: false,
      isWebGL2: false
    };

    // Remove canvas
    state.value.canvas?.remove();
    console.log('Visualization system disposed');
  };

  // Lifecycle
  onBeforeUnmount(() => {
    dispose();
  });

  return {
    state,
    initialize,
    updateFromSettings,
    updateGraphData: updateGraph,
    dispose
  };
}
