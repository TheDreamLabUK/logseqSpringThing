import { ref, computed, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useSettingsStore } from '../stores/settings';
import type { Node, Edge, CoreState, InitializationOptions } from '../types/core';
import type { PositionUpdate } from '../types/websocket';
import { POSITION_SCALE, VELOCITY_SCALE } from '../constants/websocket';

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

    // Create camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 50);
    console.debug('Camera initialized:', {
      fov: camera.fov,
      aspect: camera.aspect,
      position: camera.position.toArray()
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);
    console.debug('Lights added:', {
      ambient: { color: ambientLight.color.getHexString(), intensity: ambientLight.intensity },
      directional: { color: directionalLight.color.getHexString(), intensity: directionalLight.intensity }
    });

    // Add controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 200;
    controls.minDistance = 10;
    console.debug('Controls initialized:', {
      damping: controls.dampingFactor,
      maxDistance: controls.maxDistance,
      minDistance: controls.minDistance
    });

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

      // Log performance every 100 frames
      if (Math.floor(currentTime / 1000) % 5 === 0) {
        console.debug('Render stats:', {
          fps: state.value.fps.toFixed(1),
          drawCalls: renderer.info.render.calls,
          triangles: renderer.info.render.triangles,
          geometries: renderer.info.memory.geometries,
          textures: renderer.info.memory.textures
        });
      }

      // Render scene
      renderer.render(scene, camera);
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

      console.log('Visualization system initialized successfully');

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
      } : null
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
      } : null
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
