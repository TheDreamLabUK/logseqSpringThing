import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { useSettingsStore } from '../stores/settings';
import { useGraphSystem } from './useGraphSystem';
import { useEffectsSystem } from './useEffectsSystem';
import { usePlatform } from './usePlatform';
import type { Node, Edge, Transform, CoreState, InitializationOptions } from '../types/core';

export function useVisualization() {
  const settingsStore = useSettingsStore();
  const { getPlatformInfo, initializePlatform, getState } = usePlatform();
  
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

  // Scene setup
  const initializeScene = async () => {
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.0);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
    scene.add(hemisphereLight);

    const pointLight1 = new THREE.PointLight(0xffffff, 1.0, 300);
    pointLight1.position.set(100, 100, 100);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xffffff, 1.0, 300);
    pointLight2.position.set(-100, -100, -100);
    scene.add(pointLight2);

    // Set fog
    scene.fog = new THREE.FogExp2(0x000000, settings.value.fog_density);

    state.value.scene = scene;
  };

  // Platform initialization
  const initialize = async () => {
    if (state.value.isInitialized) return;

    try {
      // Create canvas
      const canvas = document.createElement('canvas');
      document.body.appendChild(canvas);

      // Initialize platform (browser or quest)
      const initOptions: InitializationOptions = {
        canvas,
        scene: {
          antialias: true,
          alpha: true,
          preserveDrawingBuffer: true,
          powerPreference: 'high-performance'
        },
        performance: {
          maxFPS: 60,
          enableAdaptiveQuality: true,
          enableFrustumCulling: true
        }
      };

      await initializePlatform(initOptions);
      
      // Get platform state
      const platformState = getState();
      if (!platformState) {
        throw new Error('Failed to initialize platform state');
      }

      // Update state with platform-specific components
      state.value = {
        renderer: platformState.renderer,
        camera: platformState.camera,
        scene: platformState.scene,
        canvas: platformState.canvas,
        isInitialized: true,
        isXRSupported: platformState.isXRSupported,
        isWebGL2: platformState.isWebGL2
      };

      // Initialize scene
      await initializeScene();

      // Update settings
      updateFromSettings(settings.value);

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

  // Render loop
  const render = () => {
    if (!state.value.isInitialized) return;
    
    // Render effects
    renderEffects();
  };

  // Cleanup
  const dispose = () => {
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
  };

  // Lifecycle
  onMounted(async () => {
    await initialize();
  });

  onBeforeUnmount(() => {
    dispose();
  });

  return {
    state,
    initialize,
    updateFromSettings,
    updateGraphData,
    render,
    dispose
  };
}
