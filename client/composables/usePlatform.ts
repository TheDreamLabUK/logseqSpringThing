import { ref, onMounted, onBeforeUnmount } from 'vue';
import { platformManager, type PlatformState } from '../platform/platformManager';
import type { InitializationOptions } from '../types/core';
import type { BrowserState } from '../types/platform/browser';
import type { QuestState, XRHandedness, XRHand } from '../types/platform/quest';
import type { Camera, Group, WebGLRenderer, Scene } from 'three';

export function usePlatform() {
  const isInitialized = ref(false);
  const isLoading = ref(false);
  const error = ref<Error | null>(null);

  const initialize = async (options: InitializationOptions) => {
    isLoading.value = true;
    error.value = null;

    try {
      await platformManager.initialize(options);
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
    await platformManager.enableVR();
  };

  const disableVR = async () => {
    await platformManager.disableVR();
  };

  const isVRActive = () => platformManager.isVRActive();

  // Controller and Hand Access
  const getControllerGrip = (handedness: XRHandedness): Group | null => {
    return platformManager.getControllerGrip(handedness);
  };

  const getControllerRay = (handedness: XRHandedness): Group | null => {
    return platformManager.getControllerRay(handedness);
  };

  const getHand = (handedness: XRHandedness): XRHand | null => {
    return platformManager.getHand(handedness);
  };

  // Haptic Feedback
  const vibrate = (handedness: XRHandedness, intensity?: number, duration?: number) => {
    platformManager.vibrate(handedness, intensity, duration);
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

    // XR Methods
    enableVR,
    disableVR,
    isVRActive,

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
