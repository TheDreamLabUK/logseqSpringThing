import { ref, shallowRef } from 'vue';
import type { CoreState, PlatformCapabilities, SceneConfig, Transform } from '../types/core';
import type { BrowserState, BrowserInitOptions } from '../types/platform/browser';
import type { QuestState, QuestInitOptions, XRController, XRHand, XRHandedness, XRSession } from '../types/platform/quest';
import * as THREE from 'three';
import type { Object3D, Material, BufferGeometry, PerspectiveCamera, OrthographicCamera, Group, WebGLRenderer, Camera } from 'three';

// Import OrbitControls dynamically to avoid type conflicts
let OrbitControls: any;

export type PlatformState = BrowserState | QuestState;
type ResizeCallback = (width: number, height: number) => void;
type RenderCallback = (renderer: WebGLRenderer, scene: THREE.Scene, camera: Camera) => void;

export class PlatformManager {
  private static instance: PlatformManager;
  private state = shallowRef<PlatformState | null>(null);
  private capabilities = ref<PlatformCapabilities | null>(null);
  private platform: 'browser' | 'quest' = 'browser';
  private animationFrameId: number | null = null;
  private resizeCallbacks: Set<ResizeCallback> = new Set();
  private renderCallbacks: Set<RenderCallback> = new Set();
  private lastFrameTime = 0;

  private constructor() {
    this.detectPlatform();
    this.setupResizeHandler();
  }

  static getInstance(): PlatformManager {
    if (!PlatformManager.instance) {
      PlatformManager.instance = new PlatformManager();
    }
    return PlatformManager.instance;
  }

  private setupResizeHandler() {
    const handleResize = () => {
      const state = this.state.value;
      if (!state?.canvas || !state.renderer) return;

      const width = window.innerWidth;
      const height = window.innerHeight;
      const pixelRatio = window.devicePixelRatio;

      // Update renderer
      state.renderer.setSize(width, height);
      state.renderer.setPixelRatio(pixelRatio);

      // Update camera
      if (state.camera && 'aspect' in state.camera) {
        state.camera.aspect = width / height;
        state.camera.updateProjectionMatrix();
      }

      // Update viewport
      state.viewport = {
        width,
        height,
        pixelRatio
      };

      // Notify callbacks
      this.resizeCallbacks.forEach(callback => callback(width, height));
    };

    window.addEventListener('resize', handleResize);
  }

  onResize(callback: ResizeCallback) {
    this.resizeCallbacks.add(callback);
    return () => this.resizeCallbacks.delete(callback);
  }

  onBeforeRender(callback: RenderCallback) {
    this.renderCallbacks.add(callback);
    return () => this.renderCallbacks.delete(callback);
  }

  private async detectPlatform() {
    const isQuest = /Oculus|Quest|VR/i.test(navigator.userAgent);
    const xrSupported = 'xr' in navigator;
    const webgl2 = this.checkWebGL2Support();

    this.capabilities.value = {
      webgl2,
      xr: xrSupported,
      maxTextureSize: 4096,
      maxDrawCalls: 10000,
      gpuTier: 1
    };

    this.platform = isQuest ? 'quest' : 'browser';
  }

  private checkWebGL2Support(): boolean {
    try {
      const canvas = document.createElement('canvas');
      return !!canvas.getContext('webgl2');
    } catch {
      return false;
    }
  }

  async initialize(options: BrowserInitOptions | QuestInitOptions) {
    if (this.platform === 'quest') {
      await this.initializeQuest(options as QuestInitOptions);
    } else {
      await this.initializeBrowser(options as BrowserInitOptions);
    }

    // Initial resize
    if (this.state.value?.canvas) {
      const width = window.innerWidth;
      const height = window.innerHeight;
      this.resizeCallbacks.forEach(callback => callback(width, height));
    }
  }

  private async initializeQuest(options: QuestInitOptions) {
    const sceneConfig: SceneConfig = {
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    };

    const renderer = new THREE.WebGLRenderer({
      canvas: options.canvas,
      ...sceneConfig,
      xr: { enabled: true }
    });

    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType(options.xr?.referenceSpaceType ?? 'local-floor');

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 1.6, 3);

    const scene = new THREE.Scene();

    const controllers = new Map<XRHandedness, XRController>();
    const hands = new Map<XRHandedness, XRHand>();

    // Initialize controllers
    const handednesses: XRHandedness[] = ['left', 'right'];
    handednesses.forEach((handedness, index) => {
      const controller = renderer.xr.getController(index);
      const grip = renderer.xr.getControllerGrip(index);

      controllers.set(handedness, {
        controller,
        grip,
        ray: new THREE.Group(),
        handedness,
        targetRayMode: 'tracked-pointer',
        visible: true,
        connected: false
      });

      scene.add(controller);
      scene.add(grip);
    });

    // Initialize hands
    handednesses.forEach((handedness, index) => {
      const hand = renderer.xr.getHand(index);
      
      hands.set(handedness, {
        hand,
        joints: new Map(),
        visible: true,
        connected: false
      });

      scene.add(hand);
    });

    const transform: Transform = {
      position: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1]
    };

    this.state.value = {
      type: 'xr',
      renderer,
      camera,
      scene,
      canvas: options.canvas,
      isInitialized: true,
      isXRSupported: true,
      isWebGL2: true,
      isGPUMode: false,
      fps: 0,
      lastFrameTime: 0,
      xrSession: null,
      xrSpace: null,
      xrLayer: null,
      controllers,
      hands,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        pixelRatio: window.devicePixelRatio
      },
      transform,
      config: {
        scene: sceneConfig,
        performance: {
          targetFPS: 90,
          maxDrawCalls: 10000,
          enableStats: false
        },
        xr: {
          referenceSpaceType: options.xr?.referenceSpaceType ?? 'local-floor',
          sessionMode: options.xr?.sessionMode ?? 'immersive-vr',
          optionalFeatures: options.xr?.optionalFeatures ?? ['hand-tracking'],
          requiredFeatures: options.xr?.requiredFeatures ?? ['local-floor']
        }
      }
    };

    renderer.setAnimationLoop(this.render.bind(this));
  }

  private async initializeBrowser(options: BrowserInitOptions) {
    if (!OrbitControls) {
      const module = await import('three/examples/jsm/controls/OrbitControls.js');
      OrbitControls = module.OrbitControls;
    }

    const sceneConfig: SceneConfig = {
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    };

    const renderer = new THREE.WebGLRenderer({
      canvas: options.canvas,
      ...sceneConfig
    });

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    const scene = new THREE.Scene();

    const controls = new OrbitControls(camera, options.canvas);
    Object.assign(controls, options.controls ?? {});
    controls.enableDamping = true;

    const transform: Transform = {
      position: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1]
    };

    this.state.value = {
      type: 'browser',
      renderer,
      camera,
      scene,
      canvas: options.canvas,
      controls,
      isInitialized: true,
      isXRSupported: false,
      isWebGL2: true,
      isGPUMode: false,
      fps: 0,
      lastFrameTime: 0,
      mousePosition: new THREE.Vector2(),
      touchActive: false,
      pointerLocked: false,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        pixelRatio: window.devicePixelRatio
      },
      transform,
      config: {
        scene: sceneConfig,
        performance: {
          targetFPS: 60,
          maxDrawCalls: 10000,
          enableStats: false
        }
      }
    };

    this.startRenderLoop();
  }

  private render() {
    const state = this.state.value;
    if (!state?.renderer || !state.scene || !state.camera) return;

    const now = performance.now();
    const deltaTime = now - this.lastFrameTime;
    this.lastFrameTime = now;

    // Update FPS
    state.fps = 1000 / deltaTime;
    state.lastFrameTime = now;

    if (this.platform === 'browser') {
      const browserState = state as BrowserState;
      if (browserState.controls) {
        browserState.controls.update();
      }
    }

    // Execute render callbacks
    this.renderCallbacks.forEach(callback => {
      if (state.renderer && state.scene && state.camera) {
        callback(state.renderer, state.scene, state.camera);
      }
    });

    // Final render
    state.renderer.render(state.scene, state.camera);
  }

  private startRenderLoop() {
    const animate = () => {
      this.render();
      this.animationFrameId = requestAnimationFrame(animate);
    };
    animate();
  }

  getState(): PlatformState | null {
    return this.state.value;
  }

  getCapabilities(): PlatformCapabilities | null {
    return this.capabilities.value;
  }

  getPlatform(): 'browser' | 'quest' {
    return this.platform;
  }

  isQuest(): boolean {
    return this.platform === 'quest';
  }

  isBrowser(): boolean {
    return this.platform === 'browser';
  }

  hasXRSupport(): boolean {
    return !!this.capabilities.value?.xr;
  }

  dispose() {
    if (this.state.value) {
      if (this.animationFrameId !== null) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }

      // Clear callbacks
      this.resizeCallbacks.clear();
      this.renderCallbacks.clear();

      if (this.isQuest()) {
        const questState = this.state.value as QuestState;
        if (questState.xrSession) {
          questState.xrSession.end().catch(console.error);
        }
        questState.controllers.forEach(controller => {
          controller.grip.removeFromParent();
          controller.controller.removeFromParent();
          controller.ray.removeFromParent();
          if (controller.model) controller.model.removeFromParent();
        });
        questState.hands.forEach(hand => {
          hand.hand.removeFromParent();
          if (hand.model) hand.model.removeFromParent();
        });
      } else {
        const browserState = this.state.value as BrowserState;
        if (browserState.controls) {
          browserState.controls.dispose();
        }
      }

      if (this.state.value.renderer) {
        this.state.value.renderer.dispose();
        this.state.value.renderer.forceContextLoss();
      }

      if (this.state.value.scene) {
        this.state.value.scene.traverse((object: Object3D) => {
          if (object instanceof THREE.Mesh) {
            if (object.geometry) {
              (object.geometry as BufferGeometry).dispose();
            }
            if (Array.isArray(object.material)) {
              object.material.forEach((material: Material) => material.dispose());
            } else if (object.material) {
              (object.material as Material).dispose();
            }
          }
        });
        this.state.value.scene.clear();
      }
    }

    this.state.value = null;
  }
}

export const platformManager = PlatformManager.getInstance();
