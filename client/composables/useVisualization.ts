import { ref, computed, onBeforeUnmount, provide } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useSettingsStore } from '../stores/settings';
import type { Node, Edge, CoreState, InitializationOptions } from '../types/core';
import type { PositionUpdate } from '../types/websocket';
import { POSITION_SCALE, VELOCITY_SCALE } from '../constants/websocket';

// Symbol for providing scene to components
export const SCENE_KEY = Symbol('three-scene');

export function useVisualization() {
  const settingsStore = useSettingsStore();
  
  // Core visualization state
  const state = ref<CoreState>({
    renderer: null,
    camera: null,
    scene: null,
    canvas: null,
    isInitialized: false,
    isXRSupported: false,
    isWebGL2: false,
    isGPUMode: false,
    fps: 0,
    lastFrameTime: 0
  });

  // Track animation frame for cleanup
  let animationFrameId: number | null = null;
  let controls: OrbitControls | null = null;

  // Initialize Three.js scene
  const initScene = (canvas: HTMLCanvasElement) => {
    console.debug('Initializing Three.js scene...');

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.Fog(0x000000, 50, 200);

    // Initialize scene userData
    scene.userData = {
      needsRender: true,
      lastUpdate: performance.now()
    };

    // Create camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 30, 50); // Position camera higher and back
    camera.lookAt(0, 0, 0); // Look at origin
    
    console.debug('Camera initialized:', {
      fov: camera.fov,
      aspect: camera.aspect,
      position: camera.position.toArray(),
      lookAt: [0, 0, 0]
    });

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      logarithmicDepthBuffer: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    console.debug('Renderer initialized:', {
      size: [renderer.domElement.width, renderer.domElement.height],
      pixelRatio: renderer.getPixelRatio(),
      capabilities: {
        isWebGL2: renderer.capabilities.isWebGL2,
        maxTextures: renderer.capabilities.maxTextures,
        precision: renderer.capabilities.precision
      }
    });

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 500;
    directionalLight.shadow.camera.left = -100;
    directionalLight.shadow.camera.right = 100;
    directionalLight.shadow.camera.top = 100;
    directionalLight.shadow.camera.bottom = -100;

    // Add light target
    const lightTarget = new THREE.Object3D();
    lightTarget.position.set(0, 0, 0);
    scene.add(lightTarget);
    directionalLight.target = lightTarget;
    scene.add(directionalLight);
    
    // Add hemisphere light for better ambient illumination
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);
    
    console.debug('Lights added:', {
      ambient: { color: ambientLight.color.getHexString(), intensity: ambientLight.intensity },
      directional: { 
        color: directionalLight.color.getHexString(), 
        intensity: directionalLight.intensity,
        position: directionalLight.position.toArray(),
        target: directionalLight.target.position.toArray()
      },
      hemisphere: {
        skyColor: hemiLight.color.getHexString(),
        groundColor: (hemiLight as THREE.HemisphereLight).groundColor.getHexString(),
        intensity: hemiLight.intensity
      },
      sceneChildren: scene.children.length
    });

    // Add controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 200;
    controls.minDistance = 10;
    controls.maxPolarAngle = Math.PI * 0.8; // Prevent camera from going below ground
    controls.target.set(0, 0, 0); // Look at origin
    
    console.debug('Controls initialized:', {
      damping: controls.dampingFactor,
      maxDistance: controls.maxDistance,
      minDistance: controls.minDistance,
      target: controls.target.toArray()
    });

    // Provide scene to components
    provide(SCENE_KEY, scene);
    return { scene, camera, renderer };
  };

  // Animation loop
  const animate = () => {
    if (!state.value.isInitialized) return;

    const { renderer, scene, camera } = state.value;
    if (renderer && scene && camera) {
      // Update controls
      controls?.update();

      // Update FPS counter
      const currentTime = performance.now();
      const delta = currentTime - state.value.lastFrameTime;
      state.value.fps = 1000 / delta;
      state.value.lastFrameTime = currentTime;

      // Check if scene needs render
      const needsRender = scene.userData?.needsRender !== false || 
                         controls?.enabled || 
                         currentTime - (scene.userData?.lastUpdate || 0) > 1000;

      if (needsRender) {
        // Render scene
        renderer.render(scene, camera);
        
        // Update stats
        if (process.env.NODE_ENV === 'development' && Math.floor(currentTime / 1000) % 5 === 0) {
          console.debug('Render stats:', {
            fps: state.value.fps.toFixed(1),
            drawCalls: renderer.info.render.calls,
            triangles: renderer.info.render.triangles,
            geometries: renderer.info.memory.geometries,
            textures: renderer.info.memory.textures,
            sceneChildren: scene.children.length,
            cameraPosition: camera.position.toArray(),
            controlsTarget: controls?.target.toArray()
          });
        }

        // Reset render flag and update timestamp
        scene.userData.needsRender = false;
        scene.userData.lastUpdate = currentTime;
      }
    }

    animationFrameId = requestAnimationFrame(animate);
  };

  // Initialize visualization system
  const initialize = async (options: InitializationOptions) => {
    if (state.value.isInitialized) return;

    try {
      console.log('Initializing visualization system...');
      const { scene, camera, renderer } = initScene(options.canvas);

      // Store state
      state.value = {
        renderer,
        camera,
        scene,
        canvas: options.canvas,
        isInitialized: true,
        isXRSupported: false,
        isWebGL2: renderer.capabilities.isWebGL2,
        isGPUMode: false,
        fps: 0,
        lastFrameTime: performance.now()
      };

      // Start animation loop
      animate();

      // Handle window resize
      const handleResize = () => {
        if (!camera || !renderer) return;
        
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        console.debug('Window resized:', {
          size: [window.innerWidth, window.innerHeight],
          aspect: camera.aspect
        });
      };
      window.addEventListener('resize', handleResize);

      console.log('Visualization system initialized successfully', {
        sceneChildren: scene.children.length,
        cameraPosition: camera.position.toArray(),
        rendererInfo: renderer.info.render
      });

    } catch (error) {
      console.error('Failed to initialize visualization:', error);
      throw error;
    }
  };

  // Handle binary position updates
  const updatePositions = (positions: PositionUpdate[], isInitialLayout: boolean) => {
    if (!state.value.scene || !state.value.isInitialized) return;

    console.debug('Updating positions:', {
      count: positions.length,
      isInitial: isInitialLayout,
      sample: positions[0] ? {
        id: positions[0].id,
        position: [positions[0].x / POSITION_SCALE, positions[0].y / POSITION_SCALE, positions[0].z / POSITION_SCALE],
        velocity: [positions[0].vx / VELOCITY_SCALE, positions[0].vy / VELOCITY_SCALE, positions[0].vz / VELOCITY_SCALE]
      } : null,
      sceneChildren: state.value.scene.children.length
    });

    // Enable GPU mode on first position update
    if (!state.value.isGPUMode) {
      state.value.isGPUMode = true;
      console.log('Switching to GPU-accelerated mode');
    }
  };

  // Update nodes
  const updateNodes = (nodes: Node[]) => {
    if (!state.value.scene) return;

    console.debug('Updating nodes:', {
      count: nodes.length,
      sample: nodes[0] ? {
        id: nodes[0].id,
        position: nodes[0].position,
        size: nodes[0].size,
        weight: nodes[0].weight
      } : null,
      sceneChildren: state.value.scene.children.length
    });
  };

  // Cleanup
  onBeforeUnmount(() => {
    console.log('Disposing visualization system...');
    
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    if (controls) {
      controls.dispose();
      controls = null;
    }

    if (!state.value.isInitialized) return;

    // Clean up renderer
    if (state.value.renderer) {
      state.value.renderer.dispose();
      state.value.renderer.forceContextLoss();
    }
    
    // Remove canvas
    state.value.canvas?.remove();

    // Reset state
    state.value = {
      renderer: null,
      camera: null,
      scene: null,
      canvas: null,
      isInitialized: false,
      isXRSupported: false,
      isWebGL2: false,
      isGPUMode: false,
      fps: 0,
      lastFrameTime: 0
    };

    console.log('Visualization system disposed');
  });

  return {
    state,
    initialize,
    updateNodes,
    updatePositions
  };
}
