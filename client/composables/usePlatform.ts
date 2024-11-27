import { ref, onMounted, onBeforeUnmount } from 'vue';
import { platformManager, type PlatformState } from '../platform/platformManager';
import type { SceneConfig } from '../types/core';
import type { BrowserState, BrowserInitOptions } from '../types/platform/browser';
import type { QuestState, QuestInitOptions, XRHandedness, XRHand } from '../types/platform/quest';
import type { Camera, Group, WebGLRenderer, Scene } from 'three';

// Convert core initialization options to platform-specific options
const convertToPlatformOptions = (options: {
  canvas: HTMLCanvasElement;
  scene?: Partial<SceneConfig>;
}): BrowserInitOptions | QuestInitOptions => {
  const baseOptions = {
    canvas: options.canvas,
    scene: options.scene ? {
      antialias: options.scene.antialias ?? true,
      alpha: options.scene.alpha ?? true,
      preserveDrawingBuffer: options.scene.preserveDrawingBuffer ?? true,
      powerPreference: options.scene.powerPreference ?? 'high-performance'
    } : undefined
  };

  if (platformManager.isQuest()) {
    return {
      ...baseOptions,
      xr: {
        referenceSpaceType: 'local-floor',
        sessionMode: 'immersive-vr',
        optionalFeatures: ['hand-tracking'],
        requiredFeatures: ['local-floor']
      }
    } as QuestInitOptions;
  }

  return baseOptions as BrowserInitOptions;
};

export function usePlatform() {
  const isInitialized = ref(false);
  const isLoading = ref(false);
  const error = ref<Error | null>(null);

  const initialize = async (options: { canvas: HTMLCanvasElement; scene?: Partial<SceneConfig> }) => {
    isLoading.value = true;
    error.value = null;

    try {
      const platformOptions = convertToPlatformOptions(options);
      await platformManager.initialize(platformOptions);
      isInitialized.value = true;
    } catch (err) {
      error.value = err instanceof Error ? err : new Error('Failed to initialize platform');
      console.error('Platform initialization failed:', err);
    } finally {
      isLoading.value = false;
    }
  };

  // Alias for backward compatibility
  const initializePlatform = initialize;

  const getState = <T extends PlatformState>(): T | null => {
    return platformManager.getState() as T | null;
  };

  const getBrowserState = (): BrowserState | null => {
    if (!platformManager.isBrowser()) return null;
    return getState<BrowserState>();
  };

  const getQuestState = (): QuestState | null => {
    if (!platformManager.isQuest()) return null;
    return getState<QuestState>();
  };

  const getPlatformInfo = () => {
    return {
      platform: platformManager.getPlatform(),
      capabilities: platformManager.getCapabilities(),
      isQuest: platformManager.isQuest(),
      isBrowser: platformManager.isBrowser(),
      hasXRSupport: platformManager.hasXRSupport()
    };
  };

  // XR Session Management
  const enableVR = async () => {
    if (!platformManager.hasXRSupport()) {
      throw new Error('WebXR not supported');
    }
    // Since we've removed VR support, this is a no-op
    console.warn('VR support has been removed');
  };

  const disableVR = async () => {
    // Since we've removed VR support, this is a no-op
    console.warn('VR support has been removed');
  };

  const isVRActive = () => false; // VR support removed

  // Controller and Hand Access
  const getControllerGrip = (handedness: XRHandedness): Group | null => {
    return null; // VR support removed
  };

  const getControllerRay = (handedness: XRHandedness): Group | null => {
    return null; // VR support removed
  };

  const getHand = (handedness: XRHandedness): XRHand | null => {
    return null; // VR support removed
  };

  // Haptic Feedback
  const vibrate = (handedness: XRHandedness, intensity?: number, duration?: number) => {
    // VR support removed
    console.warn('VR support has been removed');
  };

  // Render and Resize Callbacks
  const onResize = (callback: (width: number, height: number) => void) => {
    return platformManager.onResize(callback);
  };

  const onBeforeRender = (callback: (renderer: WebGLRenderer, scene: Scene, camera: Camera) => void) => {
    return platformManager.onBeforeRender(callback);
  };

  // Lifecycle
  onMounted(() => {
    // Platform manager handles resize internally
  });

  onBeforeUnmount(() => {
    if (isInitialized.value) {
      platformManager.dispose();
    }
  });

  return {
    // State
    isInitialized,
    isLoading,
    error,

    // Core Methods
    initialize,
    initializePlatform, // Backward compatibility
    getState,
    getBrowserState,
    getQuestState,
    getPlatformInfo,

    // Platform Checks
    isQuest: platformManager.isQuest,
    isBrowser: platformManager.isBrowser,
    hasXRSupport: platformManager.hasXRSupport,

    // XR Methods (now stubs since VR support is removed)
    enableVR,
    disableVR,
    isVRActive,

    // Controller Methods (now stubs since VR support is removed)
    getControllerGrip,
    getControllerRay,
    getHand,
    vibrate,

    // Event Callbacks
    onResize,
    onBeforeRender
  };
}
