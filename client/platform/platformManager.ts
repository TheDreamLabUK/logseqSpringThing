import { ref, shallowRef } from 'vue';
import type { CoreState, PlatformCapabilities, SceneConfig, Transform } from '../types/core';
import type { BrowserState, BrowserInitOptions } from '../types/platform/browser';
import type { QuestState, QuestInitOptions, XRController, XRHand, XRHandedness, XRSession, XRHitTestSource } from '../types/platform/quest';
import * as THREE from 'three';
import type { Object3D, Material, BufferGeometry, PerspectiveCamera, OrthographicCamera, Group, WebGLRenderer, Camera } from 'three';

// Import OrbitControls dynamically to avoid type conflicts
let OrbitControls: any;

export type PlatformState = BrowserState | QuestState;
type ResizeCallback = (width: number, height: number) => void;
type RenderCallback = (renderer: WebGLRenderer, scene: THREE.Scene, camera: Camera) => void;

/**
 * Platform Manager Interface
 * Defines the contract for platform-specific implementations
 */
export interface IPlatformManager {
  initialize(options: BrowserInitOptions | QuestInitOptions): Promise<void>;
  dispose(): void;
  getState(): PlatformState | null;
  getCapabilities(): PlatformCapabilities | null;
  getPlatform(): 'browser' | 'quest';
  isQuest(): boolean;
  isBrowser(): boolean;
  hasXRSupport(): boolean;
  startXRSession(mode: 'immersive-vr' | 'immersive-ar'): Promise<XRSession>;
  endXRSession(): Promise<void>;
  isInXRSession(): boolean;
  getXRSessionMode(): 'immersive-vr' | 'immersive-ar' | null;
  onResize(callback: ResizeCallback): () => void;
  onBeforeRender(callback: RenderCallback): () => void;
}

/**
 * Platform Manager Implementation
 * Handles platform-specific initialization and XR session management
 */
export class PlatformManager implements IPlatformManager {
  private static instance: PlatformManager;
  private state = shallowRef<PlatformState | null>(null);
  private capabilities = ref<PlatformCapabilities | null>(null);
  private platform: 'browser' | 'quest' = 'browser';
  private animationFrameId: number | null = null;
  private resizeCallbacks: Set<ResizeCallback> = new Set();
  private renderCallbacks: Set<RenderCallback> = new Set();
  private lastFrameTime = 0;
  private xrSessionMode: 'immersive-vr' | 'immersive-ar' | null = null;

  private constructor() {
    this.detectPlatform();
    this.setupResizeHandler();
  }

  /**
   * Get singleton instance of PlatformManager
   */
  static getInstance(): PlatformManager {
    if (!PlatformManager.instance) {
      PlatformManager.instance = new PlatformManager();
    }
    return PlatformManager.instance;
  }

  /**
   * Initialize platform with given options
   */
  async initialize(options: BrowserInitOptions | QuestInitOptions): Promise<void> {
    if (this.platform === 'quest') {
      await this.initializeQuest(options as QuestInitOptions);
    } else {
      await this.initializeBrowser(options as BrowserInitOptions);
    }

    if (this.state.value?.canvas) {
      const width = window.innerWidth;
      const height = window.innerHeight;
      this.resizeCallbacks.forEach(callback => callback(width, height));
    }
  }

  /**
   * Initialize Quest platform with XR support
   */
  private async initializeQuest(options: QuestInitOptions): Promise<void> {
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
      hitTestSource: null,
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

  /**
   * Initialize browser platform with standard WebGL support
   */
  private async initializeBrowser(options: BrowserInitOptions): Promise<void> {
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

  /**
   * Start XR session with specified mode (VR or AR)
   */
  async startXRSession(mode: 'immersive-vr' | 'immersive-ar'): Promise<XRSession> {
    const state = this.state.value;
    if (!state?.renderer) {
      throw new Error('Renderer not initialized');
    }

    try {
      if (!navigator.xr) {
        throw new Error('WebXR not supported');
      }

      const isSupported = await navigator.xr.isSessionSupported(mode);
      if (!isSupported) {
        throw new Error(`${mode} not supported`);
      }

      // Configure session features based on mode
      const sessionInit: XRSessionInit = {
        optionalFeatures: [],
        requiredFeatures: []
      };

      // Base features for both modes
      sessionInit.optionalFeatures = [
        'local-floor',
        'bounded-floor',
        'hand-tracking',
        'layers'
      ];

      // Add AR-specific features
      if (mode === 'immersive-ar') {
        sessionInit.optionalFeatures.push(
          'dom-overlay',
          'hit-test',
          'anchors',
          'plane-detection',
          'light-estimation'
        );
        sessionInit.requiredFeatures = ['hit-test'];
        sessionInit.domOverlay = { root: document.body };
      } else {
        // VR-specific features
        sessionInit.requiredFeatures = ['local-floor'];
      }

      const session = await navigator.xr.requestSession(mode, sessionInit);
      if (!session) {
        throw new Error('Failed to create XR session');
      }

      // Set up XR layer
      const gl = state.renderer.getContext();
      const xrLayer = new XRWebGLLayer(session, gl);
      await session.updateRenderState({ baseLayer: xrLayer });

      // Set reference space based on mode
      const referenceSpaceType = mode === 'immersive-ar' ? 'unbounded' : 'local-floor';
      const referenceSpace = await session.requestReferenceSpace(referenceSpaceType);

      // Set up hit testing for AR
      if (mode === 'immersive-ar' && 'requestHitTestSource' in session) {
        try {
          const hitTestSourcePromise = session.requestHitTestSource?.({
            space: referenceSpace
          });

          if (hitTestSourcePromise) {
            const hitTestSource = await hitTestSourcePromise;
            if (this.state.value && 'xrSession' in this.state.value) {
              this.state.value.hitTestSource = hitTestSource;
            }
          }
        } catch (error) {
          console.warn('Hit testing not available:', error);
        }
      }

      if (this.state.value && 'xrSession' in this.state.value) {
        this.state.value.xrSession = session;
        this.state.value.xrSpace = referenceSpace;
        this.state.value.xrLayer = xrLayer;
      }

      this.xrSessionMode = mode;

      // Set up session end handler
      session.addEventListener('end', () => {
        this.xrSessionMode = null;
        if (this.state.value && 'xrSession' in this.state.value) {
          this.state.value.xrSession = null;
          this.state.value.xrSpace = null;
          this.state.value.xrLayer = null;
          this.state.value.hitTestSource = null;
        }
      });

      // Update renderer and camera for AR
      if (mode === 'immersive-ar') {
        state.renderer.xr.setReferenceSpaceType('unbounded');
        const camera = state.camera;
        if (camera && 'aspect' in camera) {
          camera.near = 0.01;
          camera.far = 1000;
          camera.updateProjectionMatrix();
        }
      }

      return session;

    } catch (error) {
      console.error(`Error starting ${mode} session:`, error);
      throw error;
    }
  }

  /**
   * End current XR session
   */
  async endXRSession(): Promise<void> {
    const state = this.state.value;
    if (state && 'xrSession' in state && state.xrSession) {
      await state.xrSession.end();
    }
  }

  /**
   * Get current platform state
   */
  getState(): PlatformState | null {
    return this.state.value;
  }

  /**
   * Get platform capabilities
   */
  getCapabilities(): PlatformCapabilities | null {
    return this.capabilities.value;
  }

  /**
   * Get current platform type
   */
  getPlatform(): 'browser' | 'quest' {
    return this.platform;
  }

  /**
   * Check if current platform is Quest
   */
  isQuest(): boolean {
    return this.platform === 'quest';
  }

  /**
   * Check if current platform is browser
   */
  isBrowser(): boolean {
    return this.platform === 'browser';
  }

  /**
   * Check if XR is supported
   */
  hasXRSupport(): boolean {
    return !!this.capabilities.value?.xr;
  }

  /**
   * Check if XR session is active
   */
  isInXRSession(): boolean {
    return this.xrSessionMode !== null;
  }

  /**
   * Get current XR session mode
   */
  getXRSessionMode(): 'immersive-vr' | 'immersive-ar' | null {
    return this.xrSessionMode;
  }

  /**
   * Add resize callback
   */
  onResize(callback: ResizeCallback): () => void {
    this.resizeCallbacks.add(callback);
    return () => this.resizeCallbacks.delete(callback);
  }

  /**
   * Add render callback
   */
  onBeforeRender(callback: RenderCallback): () => void {
    this.renderCallbacks.add(callback);
    return () => this.renderCallbacks.delete(callback);
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.state.value) {
      if (this.animationFrameId !== null) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }

      this.resizeCallbacks.clear();
      this.renderCallbacks.clear();

      if (this.isQuest()) {
        const questState = this.state.value as QuestState;
        if (questState.xrSession) {
          questState.xrSession.end().catch(console.error);
        }
      }

      if (this.state.value.renderer) {
        this.state.value.renderer.dispose();
        this.state.value.renderer.forceContextLoss();
      }

      this.state.value = null;
    }
  }

  /**
   * Detect platform and capabilities
   */
  private async detectPlatform() {
    const isQuest = /Oculus|Quest|VR/i.test(navigator.userAgent);
    const xrSupported = 'xr' in navigator;
    const webgl2 = this.checkWebGL2Support();

    // Check AR and VR support
    let arSupported = false;
    let vrSupported = false;
    
    if (xrSupported && navigator.xr) {
      try {
        arSupported = await navigator.xr.isSessionSupported('immersive-ar');
        vrSupported = await navigator.xr.isSessionSupported('immersive-vr');
      } catch (error) {
        console.warn('Error checking XR support:', error);
      }
    }

    this.capabilities.value = {
      webgl2,
      xr: xrSupported,
      ar: arSupported,
      vr: vrSupported,
      maxTextureSize: 4096,
      maxDrawCalls: 10000,
      gpuTier: 1
    };

    this.platform = isQuest ? 'quest' : 'browser';
  }

  /**
   * Check WebGL2 support
   */
  private checkWebGL2Support(): boolean {
    try {
      const canvas = document.createElement('canvas');
      return !!canvas.getContext('webgl2');
    } catch {
      return false;
    }
  }

  /**
   * Set up resize handler
   */
  private setupResizeHandler() {
    const handleResize = () => {
      const state = this.state.value;
      if (!state?.canvas || !state.renderer) return;

      const width = window.innerWidth;
      const height = window.innerHeight;
      const pixelRatio = window.devicePixelRatio;

      state.renderer.setSize(width, height);
      state.renderer.setPixelRatio(pixelRatio);

      if (state.camera && 'aspect' in state.camera) {
        state.camera.aspect = width / height;
        state.camera.updateProjectionMatrix();
      }

      state.viewport = {
        width,
        height,
        pixelRatio
      };

      this.resizeCallbacks.forEach(callback => callback(width, height));
    };

    window.addEventListener('resize', handleResize);
  }

  /**
   * Render loop
   */
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

  /**
   * Start render loop
   */
  private startRenderLoop() {
    const animate = () => {
      this.render();
      this.animationFrameId = requestAnimationFrame(animate);
    };
    animate();
  }
}

export const platformManager = PlatformManager.getInstance();
