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
    return platformManager.startXRSession('immersive-vr');
  };

  const enableAR = async () => {
    if (!platformManager.hasXRSupport()) {
      throw new Error('WebXR not supported');
    }
    const capabilities = platformManager.getCapabilities();
    if (!capabilities?.ar) {
      throw new Error('AR not supported on this device');
    }
    return platformManager.startXRSession('immersive-ar');
  };

  const disableXR = async () => {
    return platformManager.endXRSession();
  };

  const isXRActive = () => platformManager.isInXRSession();
  const isVRActive = () => platformManager.getXRSessionMode() === 'immersive-vr';
  const isARActive = () => platformManager.getXRSessionMode() === 'immersive-ar';

  // Controller and Hand Access
  const getControllerGrip = (handedness: XRHandedness): Group | null => {
    const state = getQuestState();
    return state?.controllers.get(handedness)?.grip ?? null;
  };

  const getControllerRay = (handedness: XRHandedness): Group | null => {
    const state = getQuestState();
    return state?.controllers.get(handedness)?.ray ?? null;
  };

  const getHand = (handedness: XRHandedness): XRHand | null => {
    const state = getQuestState();
    return state?.hands.get(handedness) ?? null;
  };

  // Haptic Feedback
  const vibrate = (handedness: XRHandedness, intensity = 1.0, duration = 100) => {
    const state = getQuestState();
    const controller = state?.controllers.get(handedness);
    if (controller?.gamepad?.hapticActuators?.[0]) {
      controller.gamepad.hapticActuators[0].pulse(intensity, duration);
    }
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
    initializePlatform,
    getState,
    getBrowserState,
    getQuestState,
    getPlatformInfo,

    // Platform Checks
    isQuest: platformManager.isQuest,
    isBrowser: platformManager.isBrowser,
    hasXRSupport: platformManager.hasXRSupport,

    // XR Methods
    enableVR,
    enableAR,
    disableXR,
    isXRActive,
    isVRActive,
    isARActive,

    // Controller Methods
    getControllerGrip,
    getControllerRay,
    getHand,
    vibrate,

    // Event Callbacks
    onResize,
    onBeforeRender
  };
}
