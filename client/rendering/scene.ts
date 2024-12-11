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
import { createLogger } from '../core/utils';
import { platformManager } from '../platform/platformManager';
import { settingsManager } from '../state/settings';

const logger = createLogger('SceneManager');

// Center point between the two nodes
const SCENE_CENTER = new THREE.Vector3(-8.27, 8.11, 2.82);

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
    logger.log('Initializing SceneManager');
    
    // Initialize viewport
    this.viewport = {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio
    };

    logger.log(`Viewport: ${this.viewport.width}x${this.viewport.height} (${this.viewport.devicePixelRatio})`);

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
    // Look at the center point between nodes
    this.camera.lookAt(SCENE_CENTER);
    logger.log(`Camera position: ${JSON.stringify(CAMERA_POSITION)}`);
    logger.log(`Looking at: ${JSON.stringify(SCENE_CENTER)}`);

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true
    });
    this.renderer.setSize(this.viewport.width, this.viewport.height);
    this.renderer.setPixelRatio(this.viewport.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.5;

    // Create controls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    // Set controls target to center point between nodes
    this.controls.target.copy(SCENE_CENTER);
    logger.log('Controls initialized');

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

    // Add helpers
    this.setupHelpers();

    // Setup lighting
    this.setupLighting();

    // Setup event listeners
    this.setupEventListeners();

    // Apply initial settings
    this.applySettings(settingsManager.getSettings());

    // Subscribe to settings changes
    settingsManager.subscribe(settings => this.applySettings(settings));

    logger.log('SceneManager initialization complete');
  }

  static getInstance(canvas: HTMLCanvasElement): SceneManager {
    if (!SceneManager.instance) {
      SceneManager.instance = new SceneManager(canvas);
    }
    return SceneManager.instance;
  }

  private setupHelpers(): void {
    // Add grid helper centered at the scene center
    const gridHelper = new THREE.GridHelper(100, 100, 0x444444, 0x222222);
    gridHelper.position.copy(SCENE_CENTER);
    this.scene.add(gridHelper);

    // Add axes helper at scene center
    const axesHelper = new THREE.AxesHelper(50);
    axesHelper.position.copy(SCENE_CENTER);
    this.scene.add(axesHelper);

    logger.log('Scene helpers added');
  }

  private setupLighting(): void {
    // Ambient light - increased intensity for better base illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, 2);
    this.scene.add(ambientLight);

    // Directional lights from multiple angles for better shading
    const createDirectionalLight = (x: number, y: number, z: number, intensity: number) => {
      const light = new THREE.DirectionalLight(0xffffff, intensity);
      // Position lights relative to scene center
      light.position.copy(SCENE_CENTER).add(new THREE.Vector3(x, y, z));
      // Enable shadows for better depth perception
      light.castShadow = true;
      light.shadow.mapSize.width = 512;
      light.shadow.mapSize.height = 512;
      return light;
    };

    // Add lights from different directions for better coverage
    this.scene.add(createDirectionalLight(1, 1, 1, 2));     // Top-right-front (main light)
    this.scene.add(createDirectionalLight(-1, 1, 1, 1));    // Top-left-front (fill light)
    this.scene.add(createDirectionalLight(0, -1, 0, 0.5));  // Bottom (rim light)
    this.scene.add(createDirectionalLight(0, 0, -1, 0.5));  // Back (back light)

    // Hemisphere light for subtle ambient gradient
    const hemisphereLight = new THREE.HemisphereLight(
      0xffffff, // Sky color
      0x444444, // Ground color
      2         // Intensity
    );
    hemisphereLight.position.copy(SCENE_CENTER).add(new THREE.Vector3(0, 1, 0));
    this.scene.add(hemisphereLight);

    logger.log('Enhanced lighting setup complete');
  }

  private setupEventListeners(): void {
    window.addEventListener('resize', this.handleResize.bind(this));
    logger.log('Event listeners setup');
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

    logger.log(`Viewport resized: ${this.viewport.width}x${this.viewport.height}`);
  }

  private applySettings(settings: VisualizationSettings): void {
    // Apply bloom settings
    this.bloomPass.enabled = settings.enableBloom;
    this.bloomPass.threshold = settings.bloomThreshold;
    this.bloomPass.strength = settings.bloomIntensity;
    this.bloomPass.radius = settings.bloomRadius;

    logger.log('Settings applied');
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
    logger.log('Scene rendering started');
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    logger.log('Scene rendering stopped');
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
    logger.log(`Added object to scene: ${object.type}`);
  }

  /**
   * Remove an object from the scene
   */
  remove(object: THREE.Object3D): void {
    this.scene.remove(object);
    logger.log(`Removed object from scene: ${object.type}`);
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
    
    logger.log(`Cleared ${objectsToRemove.length} objects from scene`);
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

    logger.log('Scene manager disposed');
  }
}
