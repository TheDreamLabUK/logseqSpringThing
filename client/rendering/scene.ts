/**
 * Three.js scene management with simplified setup
 */

import { Scene, PerspectiveCamera, WebGLRenderer, Color, AmbientLight, DirectionalLight, GridHelper, Vector2, Material, Mesh, Object3D } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { createLogger } from '../core/utils';
import { Settings } from '../types/settings';
import { VisualizationController } from './VisualizationController';

const logger = createLogger('SceneManager');

// Constants
const BACKGROUND_COLOR = 0x212121;  // Material Design Grey 900

export class SceneManager {
  private static instance: SceneManager;
  
  // Three.js core components
  private scene: Scene;
  private camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private sceneGrid: GridHelper | null = null;
  
  // Post-processing
  private composer: EffectComposer;
  private bloomPass: UnrealBloomPass;
  
  // Animation
  private animationFrameId: number | null = null;
  private isRunning: boolean = false;
  private visualizationController: VisualizationController | null = null;

  private constructor(canvas: HTMLCanvasElement) {
    logger.log('Initializing SceneManager');
    
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
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      xr: {
        enabled: true
      }
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create controls
    this.controls = new OrbitControls(this.camera, canvas);
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
    this.composer = new EffectComposer(this.renderer);
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    // Initialize bloom with default state
    this.bloomPass = new UnrealBloomPass(
      new Vector2(window.innerWidth, window.innerHeight),
      1.5,  // Default strength
      0.4,  // Default radius
      0.85  // Default threshold
    );
    
    // Initialize custom bloom properties
    (this.bloomPass as any).edgeStrength = 3.0;
    (this.bloomPass as any).nodeStrength = 2.0;
    (this.bloomPass as any).environmentStrength = 1.0;
    
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
    this.animate();
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

  private animate = (): void => {
    if (!this.isRunning) return;

    // Set up animation loop
    if (this.renderer.xr.enabled) {
      // For XR, use the built-in animation loop
      this.renderer.setAnimationLoop(this.render);
    } else {
      // For non-XR, use requestAnimationFrame
      this.animationFrameId = requestAnimationFrame(this.animate);
      this.render();
    }
  }

  private render = (): void => {
    // Update controls only in non-XR mode
    if (!this.renderer.xr.enabled) {
      this.controls.update();
      // Show scene grid in non-XR mode
      if (this.sceneGrid) this.sceneGrid.visible = true;
    } else {
      // Hide scene grid in XR mode
      if (this.sceneGrid) this.sceneGrid.visible = false;
    }

    // Update visualization controller
    this.visualizationController?.update();

    // Use post-processing in non-XR mode when bloom is enabled
    if (!this.renderer.xr.enabled && this.bloomPass.enabled) {
      this.composer.render();
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

  getControls(): OrbitControls {
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

    const { rendering, bloom } = settings.visualization;

    // Update bloom settings
    if (bloom) {
      // Always update enabled state first
      this.bloomPass.enabled = bloom.enabled;
      
      if (bloom.enabled) {
        // When enabled, set all parameters
        this.bloomPass.strength = bloom.strength || 1.5;
        this.bloomPass.radius = bloom.radius || 0.4;
        this.bloomPass.threshold = 0.3; // Lower threshold when enabled for better effect
        
        // Set custom strength parameters
        (this.bloomPass as any).edgeStrength = bloom.edgeBloomStrength || 3.0;
        (this.bloomPass as any).nodeStrength = bloom.nodeBloomStrength || 2.0;
        (this.bloomPass as any).environmentStrength = bloom.environmentBloomStrength || 1.0;
      } else {
        // When disabled, zero out all parameters
        this.bloomPass.strength = 0;
        this.bloomPass.radius = 0;
        this.bloomPass.threshold = 1.0;
        (this.bloomPass as any).edgeStrength = 0;
        (this.bloomPass as any).nodeStrength = 0;
        (this.bloomPass as any).environmentStrength = 0;
      }
      
      logger.debug('Bloom settings updated:', {
        enabled: this.bloomPass.enabled,
        strength: this.bloomPass.strength,
        radius: this.bloomPass.radius,
        threshold: this.bloomPass.threshold,
        edgeStrength: (this.bloomPass as any).edgeStrength,
        nodeStrength: (this.bloomPass as any).nodeStrength,
        environmentStrength: (this.bloomPass as any).environmentStrength
      });
    }

    // Update background color
    if (rendering.backgroundColor) {
      this.scene.background = new Color(rendering.backgroundColor);
    }

    // Update lighting
    const lights = this.scene.children.filter(child => 
      child instanceof AmbientLight || child instanceof DirectionalLight
    );
    
    lights.forEach(light => {
      if (light instanceof AmbientLight) {
        light.intensity = rendering.ambientLightIntensity;
      } else if (light instanceof DirectionalLight) {
        light.intensity = rendering.directionalLightIntensity;
      }
    });

    // Update renderer settings
    if (this.renderer) {
      // Note: Some settings can only be changed at renderer creation
      if (rendering.enableAntialiasing) {
        logger.warn('Antialiasing setting change requires renderer recreation');
      }
      if (rendering.enableShadows) {
        logger.warn('Shadow settings change requires renderer recreation');
      }
    }

    logger.debug('Scene settings updated:', rendering);
  }
}
