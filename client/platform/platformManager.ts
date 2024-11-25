import { ref, shallowRef } from 'vue';
import type { CoreState, PlatformCapabilities } from '../types/core';
import type { BrowserState, BrowserInitOptions } from '../types/platform/browser';
import type { QuestState, QuestInitOptions, XRController, XRHand, XRHandedness, XRSession } from '../types/platform/quest';
import * as THREE from 'three';
import type { Object3D, Material, BufferGeometry, Camera, Group, WebGLRenderer } from 'three';

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
      if (!this.state.value?.canvas) return;

      const width = window.innerWidth;
      const height = window.innerHeight;
      const pixelRatio = window.devicePixelRatio;

      // Update renderer
      this.state.value.renderer.setSize(width, height);
      this.state.value.renderer.setPixelRatio(pixelRatio);

      // Update camera
      if ('aspect' in this.state.value.camera) {
        this.state.value.camera.aspect = width / height;
        this.state.value.camera.updateProjectionMatrix();
      }

      // Update viewport
      this.state.value.viewport = {
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
      multiview: this.checkMultiviewSupport(),
      instancedArrays: true,
      floatTextures: this.checkFloatTextureSupport(),
      depthTexture: true,
      drawBuffers: true,
      shaderTextureLOD: this.checkShaderTextureLODSupport()
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

  private checkMultiviewSupport(): boolean {
    const gl = document.createElement('canvas').getContext('webgl2');
    return gl ? !!gl.getExtension('OVR_multiview2') : false;
  }

  private checkFloatTextureSupport(): boolean {
    const gl = document.createElement('canvas').getContext('webgl2');
    return gl ? !!gl.getExtension('EXT_color_buffer_float') : false;
  }

  private checkShaderTextureLODSupport(): boolean {
    const gl = document.createElement('canvas').getContext('webgl2');
    return gl ? !!gl.getExtension('EXT_shader_texture_lod') : false;
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
    const renderer = new THREE.WebGLRenderer({
      canvas: options.canvas,
      antialias: options.scene?.antialias ?? true,
      alpha: options.scene?.alpha ?? true,
      powerPreference: options.scene?.powerPreference ?? 'high-performance',
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

    this.state.value = {
      renderer,
      camera: camera as Camera,
      scene,
      canvas: options.canvas,
      isInitialized: true,
      isXRSupported: true,
      isWebGL2: true,
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
      transform: {
        position: new THREE.Vector3(),
        rotation: new THREE.Vector3(),
        scale: new THREE.Vector3(1, 1, 1)
      },
      config: {
        scene: options.scene ?? {},
        performance: options.performance ?? {},
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

    const renderer = new THREE.WebGLRenderer({
      canvas: options.canvas,
      antialias: options.scene?.antialias ?? true,
      alpha: options.scene?.alpha ?? true,
      powerPreference: options.scene?.powerPreference ?? 'high-performance'
    });

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    const scene = new THREE.Scene();

    const controls = new OrbitControls(camera, options.canvas);
    Object.assign(controls, options.controls ?? {});
    controls.enableDamping = true;

    this.state.value = {
      renderer,
      camera: camera as Camera,
      scene,
      canvas: options.canvas,
      controls,
      isInitialized: true,
      isXRSupported: false,
      isWebGL2: true,
      mousePosition: new THREE.Vector2(),
      touchActive: false,
      pointerLocked: false,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        pixelRatio: window.devicePixelRatio
      },
      transform: {
        position: new THREE.Vector3(),
        rotation: new THREE.Vector3(),
        scale: new THREE.Vector3(1, 1, 1)
      },
      config: {
        scene: options.scene ?? {},
        performance: options.performance ?? {}
      }
    };

    this.startRenderLoop();
  }

  // Quest-specific methods
  async enableVR(): Promise<void> {
    if (!this.isQuest() || !this.state.value) return;

    const state = this.state.value as QuestState;
    const { sessionMode, optionalFeatures, requiredFeatures } = state.config.xr;

    try {
      const session = await (navigator as any).xr?.requestSession(sessionMode, {
        optionalFeatures,
        requiredFeatures
      });

      if (session) {
        state.xrSession = session as XRSession;
        await state.renderer.xr.setSession(session);
      }
    } catch (error) {
      console.error('Failed to start XR session:', error);
      throw error;
    }
  }

  async disableVR(): Promise<void> {
    if (!this.isQuest() || !this.state.value) return;

    const state = this.state.value as QuestState;
    if (state.xrSession) {
      await state.xrSession.end();
      state.xrSession = null;
    }
  }

  isVRActive(): boolean {
    if (!this.isQuest() || !this.state.value) return false;
    return !!(this.state.value as QuestState).xrSession;
  }

  getControllerGrip(handedness: XRHandedness): Group | null {
    if (!this.isQuest() || !this.state.value) return null;
    return (this.state.value as QuestState).controllers.get(handedness)?.grip ?? null;
  }

  getControllerRay(handedness: XRHandedness): Group | null {
    if (!this.isQuest() || !this.state.value) return null;
    return (this.state.value as QuestState).controllers.get(handedness)?.ray ?? null;
  }

  getHand(handedness: XRHandedness): XRHand | null {
    if (!this.isQuest() || !this.state.value) return null;
    return (this.state.value as QuestState).hands.get(handedness) ?? null;
  }

  vibrate(handedness: XRHandedness, intensity = 1.0, duration = 100): void {
    if (!this.isQuest() || !this.state.value) return;
    
    const controller = (this.state.value as QuestState).controllers.get(handedness);
    if (controller?.gamepad?.hapticActuators?.[0]) {
      controller.gamepad.hapticActuators[0].pulse(intensity, duration);
    }
  }

  private render() {
    if (!this.state.value) return;

    const state = this.state.value;
    if (this.platform === 'browser') {
      const browserState = state as BrowserState;
      if (browserState.controls) {
        browserState.controls.update();
      }
    }

    // Execute render callbacks
    this.renderCallbacks.forEach(callback => {
      callback(state.renderer, state.scene, state.camera);
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
