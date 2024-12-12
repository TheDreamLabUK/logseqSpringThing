/**
 * Three.js scene management with simplified setup
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { createLogger } from '../core/utils';

const logger = createLogger('SceneManager');

// Constants
const BACKGROUND_COLOR = 0x212121;  // Material Design Grey 900

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

  private constructor(canvas: HTMLCanvasElement) {
    logger.log('Initializing SceneManager');
    
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND_COLOR);
    this.scene.fog = new THREE.FogExp2(BACKGROUND_COLOR, 0.002);

    // Create camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 75, 200);
    this.camera.lookAt(0, 0, 0);

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create controls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 50;
    this.controls.maxDistance = 500;

    // Setup post-processing
    this.composer = new EffectComposer(this.renderer);
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      1.5,  // Strength
      0.75, // Radius
      0.3   // Threshold
    );
    this.composer.addPass(this.bloomPass);

    // Setup basic lighting
    this.setupLighting();

    // Setup event listeners
    window.addEventListener('resize', this.handleResize.bind(this));

    logger.log('SceneManager initialization complete');
  }

  static getInstance(canvas: HTMLCanvasElement): SceneManager {
    if (!SceneManager.instance) {
      SceneManager.instance = new SceneManager(canvas);
    }
    return SceneManager.instance;
  }

  private setupLighting(): void {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1).normalize();
    this.scene.add(directionalLight);

    // Add grid helper
    const gridHelper = new THREE.GridHelper(1000, 100);
    if (gridHelper.material instanceof THREE.Material) {
      gridHelper.material.transparent = true;
      gridHelper.material.opacity = 0.1;
    }
    this.scene.add(gridHelper);
  }

  private handleResize(): void {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(width, height);
    this.composer.setSize(width, height);
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

  private animate(): void {
    if (!this.isRunning) return;

    this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
    this.controls.update();
    this.composer.render();
  }

  // Public getters
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

  // Scene management methods
  add(object: THREE.Object3D): void {
    this.scene.add(object);
  }

  remove(object: THREE.Object3D): void {
    this.scene.remove(object);
  }

  dispose(): void {
    this.stop();
    
    window.removeEventListener('resize', this.handleResize.bind(this));

    this.renderer.dispose();
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        if (object.material instanceof THREE.Material) {
          object.material.dispose();
        }
      }
    });

    logger.log('Scene manager disposed');
  }
}
