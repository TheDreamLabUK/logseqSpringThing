/**
 * Three.js scene management and basic rendering
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';

import { Viewport, VisualizationSettings } from '../core/types';
import { CAMERA_FOV, CAMERA_NEAR, CAMERA_FAR, CAMERA_POSITION } from '../core/constants';
// import { createLogger } from '../core/utils';
import { platformManager } from '../platform/platformManager';
import { settingsManager } from '../state/settings';

// Logger will be used for debugging scene setup, rendering performance, and XR mode transitions
// import { createLogger } from '../core/utils';
// const __logger = createLogger('SceneManager');

export class SceneManager {
  private static instance: SceneManager;
  
  // Three.js core components
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  
  // Post-processing
  private composer: EffectComposer;
  private bloomPass: UnrealBloomPass;
  
  // Animation
  private animationFrameId: number | null = null;
  private isRunning: boolean = false;
  
  // Viewport
  private viewport: Viewport;
  
  private constructor(canvas: HTMLCanvasElement) {
    // Initialize viewport
    this.viewport = {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio
    };

    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    // Create camera
    this.camera = new THREE.PerspectiveCamera(
      CAMERA_FOV,
      this.viewport.width / this.viewport.height,
      CAMERA_NEAR,
      CAMERA_FAR
    );
    this.camera.position.set(
      CAMERA_POSITION.x,
      CAMERA_POSITION.y,
      CAMERA_POSITION.z
    );

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true
    });
    this.renderer.setSize(this.viewport.width, this.viewport.height);
    this.renderer.setPixelRatio(this.viewport.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Create controls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Setup post-processing
    this.composer = new EffectComposer(this.renderer);
    
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(this.viewport.width, this.viewport.height),
      1.5, // intensity
      0.4, // radius
      0.85 // threshold
    );
    this.composer.addPass(this.bloomPass);

    // Setup lighting
    this.setupLighting();

    // Setup event listeners
    this.setupEventListeners();

    // Apply initial settings
    this.applySettings(settingsManager.getSettings());

    // Subscribe to settings changes
    settingsManager.subscribe(settings => this.applySettings(settings));
  }

  static getInstance(canvas: HTMLCanvasElement): SceneManager {
    if (!SceneManager.instance) {
      SceneManager.instance = new SceneManager(canvas);
    }
    return SceneManager.instance;
  }

  private setupLighting(): void {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x404040, 1);
    this.scene.add(ambientLight);

    // Directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    this.scene.add(directionalLight);

    // Hemisphere light
    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x404040, 1);
    this.scene.add(hemisphereLight);
  }

  private setupEventListeners(): void {
    window.addEventListener('resize', this.handleResize.bind(this));
  }

  private handleResize(): void {
    this.viewport = {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio
    };

    this.camera.aspect = this.viewport.width / this.viewport.height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(this.viewport.width, this.viewport.height);
    this.renderer.setPixelRatio(this.viewport.devicePixelRatio);

    this.composer.setSize(this.viewport.width, this.viewport.height);
  }

  private applySettings(settings: VisualizationSettings): void {
    // Apply bloom settings
    this.bloomPass.enabled = settings.enableBloom;
    this.bloomPass.threshold = settings.bloomThreshold;
    this.bloomPass.strength = settings.bloomIntensity;
    this.bloomPass.radius = settings.bloomRadius;

    // Apply other visual settings as needed
  }

  private animate(): void {
    if (!this.isRunning) return;

    this.animationFrameId = requestAnimationFrame(this.animate.bind(this));

    // Update controls
    this.controls.update();

    // Render scene
    if (platformManager.isQuest()) {
      // XR rendering will be handled by XRManager
      return;
    }

    this.composer.render();
  }

  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.animate();
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  // Public API

  getScene(): THREE.Scene {
    return this.scene;
  }

  getCamera(): THREE.PerspectiveCamera {
    return this.camera;
  }

  getRenderer(): THREE.WebGLRenderer {
    return this.renderer;
  }

  getControls(): OrbitControls {
    return this.controls;
  }

  getViewport(): Viewport {
    return { ...this.viewport };
  }

  /**
   * Add an object to the scene
   */
  add(object: THREE.Object3D): void {
    this.scene.add(object);
  }

  /**
   * Remove an object from the scene
   */
  remove(object: THREE.Object3D): void {
    this.scene.remove(object);
  }

  /**
   * Clear all objects from the scene except lights and camera
   */
  clear(): void {
    const objectsToRemove = this.scene.children.filter(child => 
      !(child instanceof THREE.Light) && 
      !(child instanceof THREE.Camera)
    );
    
    objectsToRemove.forEach(object => {
      this.scene.remove(object);
    });
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    this.stop();
    
    // Remove event listeners
    window.removeEventListener('resize', this.handleResize.bind(this));

    // Dispose of Three.js resources
    this.renderer.dispose();
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        if (object.material instanceof THREE.Material) {
          object.material.dispose();
        }
      }
    });

    // Clear scene
    this.clear();
  }
}
