/**
 * Three.js scene management with simplified setup
 */

import {
  Scene,
  PerspectiveCamera,
  WebGLRenderer,
  Color,
  AmbientLight,
  DirectionalLight,
  GridHelper,
  Vector2,
  Material,
  Mesh,
  Object3D
} from 'three';
import * as OrbitControlsModule from 'three/examples/jsm/controls/OrbitControls';
import * as EffectComposerModule from 'three/examples/jsm/postprocessing/EffectComposer';
import * as RenderPassModule from 'three/examples/jsm/postprocessing/RenderPass';
import * as UnrealBloomPassModule from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { createLogger } from '../core/utils';
import { Settings } from '../types/settings';
import { defaultSettings } from '../state/defaultSettings';
import { VisualizationController } from './VisualizationController';

const logger = createLogger('SceneManager');

// Constants
const BACKGROUND_COLOR = 0x212121;  // Material Design Grey 900
const LOW_PERF_FPS_THRESHOLD = 30;  // Lower FPS threshold for low performance mode

export class SceneManager {
  private static instance: SceneManager;
  
  // Three.js core components
  private scene: Scene;
  private camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private readonly canvas: HTMLCanvasElement;
  private currentRenderingSettings: Settings['visualization']['rendering'] | null = null;
  private controls!: OrbitControlsModule.OrbitControls & { dispose: () => void };
  private sceneGrid: GridHelper | null = null;
  
  // Post-processing
  private composer: EffectComposerModule.EffectComposer;
  private bloomPass: UnrealBloomPassModule.UnrealBloomPass;
  
  // Animation
  private animationFrameId: number | null = null;
  private isRunning: boolean = false;
  private visualizationController: VisualizationController | null = null;
  private lastFrameTime: number = 0;
  private readonly FRAME_BUDGET: number = 16; // Target 60fps (1000ms/60)
  private frameCount: number = 0;
  private lastFpsUpdate: number = 0;
  private currentFps: number = 60;

  private constructor(canvas: HTMLCanvasElement) {
    logger.log('Initializing SceneManager');
    this.canvas = canvas;
    
    // Create scene
    this.scene = new Scene();
    this.scene.background = new Color(BACKGROUND_COLOR);
    // Removed fog to ensure graph visibility

    // Create camera with wider view
    this.camera = new PerspectiveCamera(
      60, // Reduced FOV for less distortion
      window.innerWidth / window.innerHeight,
      0.1,
      2000
    );
    this.camera.position.set(0, 10, 50); // Position for better overview
    this.camera.lookAt(0, 0, 0);
    
    // Enable both layers for desktop mode by default
    this.camera.layers.enable(0); // Desktop layer
    this.camera.layers.enable(1); // XR layer

    // Create renderer with WebXR support
    this.renderer = new WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      xr: {
        enabled: true
      }
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Enable performance optimizations
    (this.renderer as any).sortObjects = false;  // Disable automatic object sorting
    (this.renderer as any).physicallyCorrectLights = false;  // Disable physically correct lighting

    // Create controls
    this.controls = new OrbitControlsModule.OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 1000;
    this.controls.enableRotate = true;
    this.controls.enableZoom = true;
    this.controls.enablePan = true;
    this.controls.rotateSpeed = 1.0;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.8;

    // Setup post-processing
    this.composer = new EffectComposerModule.EffectComposer(this.renderer);
    const renderPass = new RenderPassModule.RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    const bloomSettings = defaultSettings.visualization.bloom;

    // Initialize bloom with default state
    this.bloomPass = new UnrealBloomPassModule.UnrealBloomPass(
      new Vector2(window.innerWidth, window.innerHeight),
      bloomSettings.strength,
      bloomSettings.radius,
      bloomSettings.threshold
    );

    // Initialize custom bloom properties
    (this.bloomPass as any).edgeStrength = bloomSettings.edgeBloomStrength;
    (this.bloomPass as any).nodeStrength = bloomSettings.nodeBloomStrength;
    (this.bloomPass as any).environmentStrength = bloomSettings.environmentBloomStrength;
    
    this.composer.addPass(this.bloomPass);

    // Setup basic lighting
    this.setupLighting();

    // Setup event listeners
    window.addEventListener('resize', this.handleResize.bind(this));

    // Initialize visualization controller
    this.visualizationController = VisualizationController.getInstance();
    this.visualizationController.initializeScene(this.scene, this.camera);

    logger.log('SceneManager initialization complete');
  }

  static getInstance(canvas: HTMLCanvasElement): SceneManager {
    if (!SceneManager.instance) {
      SceneManager.instance = new SceneManager(canvas);
    }
    return SceneManager.instance;
  }

  static cleanup(): void {
    if (SceneManager.instance) {
      SceneManager.instance.dispose();
      SceneManager.instance = null as any;
    }
  }

  private setupLighting(): void {
    const ambientLight = new AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1).normalize();
    this.scene.add(directionalLight);

    // Add smaller grid helper
    const gridHelper = new GridHelper(50, 50); // Reduced grid size
    if (gridHelper.material instanceof Material) {
      gridHelper.material.transparent = true;
      gridHelper.material.opacity = 0.1;
    }
    this.scene.add(gridHelper);
    this.sceneGrid = gridHelper;
  }

  private handleResize(): void {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(width, height);
    this.composer.setSize(width, height);
    
    // Update bloom resolution
    if (this.bloomPass) {
      this.bloomPass.resolution.set(width, height);
    }
  }

  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    requestAnimationFrame(this.animate);
    logger.log('Scene rendering started');
  }

  // Alias for start() to maintain compatibility with new client code
  startRendering(): void {
    this.start();
  }

  stop(): void {
    this.isRunning = false;
    
    // Clean up animation loops
    if (this.renderer.xr.enabled) {
      this.renderer.setAnimationLoop(null);
    }
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    logger.log('Scene rendering stopped');
  }

  private animate = (timestamp: number): void => {
    if (!this.isRunning) return;

    // Calculate FPS
    this.frameCount++;
    if (timestamp - this.lastFpsUpdate >= 1000) {
      this.currentFps = (this.frameCount * 1000) / (timestamp - this.lastFpsUpdate);
      this.frameCount = 0;
      this.lastFpsUpdate = timestamp;

      // Apply performance optimizations if FPS is low
      if (this.currentFps < LOW_PERF_FPS_THRESHOLD) {
        this.applyLowPerformanceOptimizations();
      }
    }

    const deltaTime = timestamp - this.lastFrameTime;
    this.lastFrameTime = timestamp;

    // Set up animation loop
    if (this.renderer.xr.enabled) {
      // For XR, use the built-in animation loop
      this.renderer.setAnimationLoop(this.render);
    } else {
      // For non-XR, use requestAnimationFrame
      this.render(deltaTime);
      if (this.isRunning) {
        this.animationFrameId = requestAnimationFrame(this.animate);
      }
    }
  }

  private render = (deltaTime?: number): void => {
    const startTime = performance.now();

    // Update controls only in non-XR mode
    if (!this.renderer.xr.enabled) {
      // Only update controls if enough time has passed
      if (!deltaTime || deltaTime >= this.FRAME_BUDGET) {
        this.controls.update();
        // Show scene grid in non-XR mode
        if (this.sceneGrid) this.sceneGrid.visible = true;
      }
    } else {
      // Hide scene grid in XR mode
      if (this.sceneGrid) this.sceneGrid.visible = false;
    }

    // Check if we have time for visualization update
    // Always update visualization to maintain smooth movement
    (this.visualizationController as any)?.update(deltaTime || 0);

    // Check remaining time for rendering
    const preRenderTime = performance.now();
    const remainingTime = this.FRAME_BUDGET - (preRenderTime - startTime);

    if (remainingTime >= 0) {
      // Use post-processing in non-XR mode when bloom is enabled
      if (!this.renderer.xr.enabled && this.bloomPass.enabled) {
        // Skip bloom if we're running low on time
        if (remainingTime >= 8) { // Give bloom half our frame budget
          this.composer.render();
        } else {
          this.renderer.render(this.scene, this.camera);
        }
      } else {
        this.renderer.render(this.scene, this.camera);
      }
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  // Public getters
  getScene(): Scene {
    return this.scene;
  }

  getCamera(): PerspectiveCamera {
    return this.camera;
  }

  getRenderer(): WebGLRenderer {
    return this.renderer;
  }

  getControls(): OrbitControlsModule.OrbitControls {
    return this.controls;
  }

  // Scene management methods
  add(object: Object3D): void {
    this.scene.add(object);
  }

  remove(object: Object3D): void {
    this.scene.remove(object);
  }

  dispose(): void {
    this.stop();
    
    // Remove event listeners
    const boundResize = this.handleResize.bind(this);
    window.removeEventListener('resize', boundResize);

    // Dispose of post-processing
    if (this.composer) {
      // Dispose of render targets
      this.composer.renderTarget1.dispose();
      this.composer.renderTarget2.dispose();
      
      // Clear passes
      this.composer.passes.length = 0;
    }

    // Dispose of bloom pass resources
    if (this.bloomPass) {
      // Dispose of any textures or materials used by the bloom pass
      if ((this.bloomPass as any).renderTargetsHorizontal) {
        (this.bloomPass as any).renderTargetsHorizontal.forEach((target: any) => {
          if (target && target.dispose) target.dispose();
        });
      }
      if ((this.bloomPass as any).renderTargetsVertical) {
        (this.bloomPass as any).renderTargetsVertical.forEach((target: any) => {
          if (target && target.dispose) target.dispose();
        });
      }
      if ((this.bloomPass as any).materialHorizontal) {
        (this.bloomPass as any).materialHorizontal.dispose();
      }
      if ((this.bloomPass as any).materialVertical) {
        (this.bloomPass as any).materialVertical.dispose();
      }
    }

    // Dispose of controls
    if (this.controls) {
      this.controls.dispose();
    }

    // Dispose of renderer and materials
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.domElement.remove();
      (this.renderer.domElement as any).width = 0;
      (this.renderer.domElement as any).height = 0;
    }

    // Dispose of scene objects
    if (this.scene) {
      this.scene.traverse((object) => {
        if (object instanceof Mesh) {
          if (object.geometry) object.geometry.dispose();
          if (object.material) {
            if (Array.isArray(object.material)) {
              object.material.forEach(material => material.dispose());
            } else {
              object.material.dispose();
            }
          }
        }
      });
    }

    logger.log('Scene manager disposed');
  }

  public handleSettingsUpdate(settings: Settings): void {
    if (!settings.visualization?.rendering) {
      logger.warn('Received settings update without visualization.rendering section');
      return;
    }

    const { rendering: newRendering, bloom: newBloom } = settings.visualization;
    const hasRenderingChanged = JSON.stringify(this.currentRenderingSettings) !== JSON.stringify(newRendering);

    // Update bloom settings
    if (newBloom) {
      const currentBloom = {
        enabled: this.bloomPass.enabled,
        strength: this.bloomPass.strength,
        radius: this.bloomPass.radius,
        threshold: this.bloomPass.threshold,
        edgeStrength: (this.bloomPass as any).edgeStrength,
        nodeStrength: (this.bloomPass as any).nodeStrength,
        environmentStrength: (this.bloomPass as any).environmentStrength
      };

      const newBloomSettings = {
        enabled: newBloom.enabled,
        strength: newBloom.enabled ? (newBloom.strength || defaultSettings.visualization.bloom.strength) : 0,
        radius: newBloom.enabled ? (newBloom.radius || defaultSettings.visualization.bloom.radius) : 0,
        threshold: newBloom.threshold, // Use threshold from settings
        edgeStrength: newBloom.enabled ? (newBloom.edgeBloomStrength || defaultSettings.visualization.bloom.edgeBloomStrength) : 0,
        nodeStrength: newBloom.enabled ? (newBloom.nodeBloomStrength || defaultSettings.visualization.bloom.nodeBloomStrength) : 0,
        environmentStrength: newBloom.enabled ? (newBloom.environmentBloomStrength || defaultSettings.visualization.bloom.environmentBloomStrength) : 0
      };

      const hasBloomChanged = JSON.stringify(currentBloom) !== JSON.stringify(newBloomSettings);
      
      if (hasBloomChanged) {
        this.bloomPass.enabled = newBloomSettings.enabled;
        this.bloomPass.strength = newBloomSettings.strength;
        this.bloomPass.radius = newBloomSettings.radius;
        this.bloomPass.threshold = newBloomSettings.threshold;
        (this.bloomPass as any).edgeStrength = newBloomSettings.edgeStrength;
        (this.bloomPass as any).nodeStrength = newBloomSettings.nodeStrength;
        (this.bloomPass as any).environmentStrength = newBloomSettings.environmentStrength;
      }
    }

    if (hasRenderingChanged) {
      this.currentRenderingSettings = newRendering;

      // Update background color
      if (newRendering.backgroundColor) {
        this.scene.background = new Color(newRendering.backgroundColor);
      }

      // Update lighting
      const lights = this.scene.children.filter(child => 
        child instanceof AmbientLight || child instanceof DirectionalLight
      );
      
      lights.forEach(light => {
        if (light instanceof AmbientLight) {
          light.intensity = newRendering.ambientLightIntensity;
        } else if (light instanceof DirectionalLight) {
          light.intensity = newRendering.directionalLightIntensity;
        }
      });

      // Update renderer settings
      if (this.renderer) {
        // Log settings changes that can't be updated at runtime
        if (newRendering.enableAntialiasing !== this.currentRenderingSettings?.enableAntialiasing) {
          logger.warn('Antialiasing setting can only be changed at renderer creation');
        }
        (this.renderer as any).shadowMap.enabled = newRendering.enableShadows || false;
      }
    }

    // Only log if something actually changed
    if (hasRenderingChanged) {
      logger.debug('Scene settings updated:', {
        rendering: newRendering,
        bloom: {
          enabled: this.bloomPass.enabled,
          strength: this.bloomPass.strength
        }
      });
    }
  }

  private applyLowPerformanceOptimizations(): void {
    // Optimize materials
    this.scene.traverse((object: Object3D) => {
      if (object instanceof Mesh) {
        const material = object.material as Material;
        if (material) {
          // Keep material features that affect visual quality
          material.needsUpdate = true;
          
          // Disable shadows
          (object as any).castShadow = (object as any).receiveShadow = false;
          
          // Force material update
          material.needsUpdate = true;
        }
      }
    });

    // Optimize renderer
    (this.renderer as any).shadowMap.enabled = false;
    
    // Only disable bloom at very low FPS
    if (this.bloomPass?.enabled && this.currentFps < 20) {
      this.bloomPass.enabled = false;
    }

    // Log optimization application
    logger.debug(`Applied low performance optimizations at ${this.currentFps.toFixed(1)} FPS`);
  }
}
