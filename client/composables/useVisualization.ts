import { ref, computed, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useSettingsStore } from '../stores/settings';
import type { Node, Edge, CoreState, InitializationOptions } from '../types/core';
import type { PositionUpdate } from '../types/websocket';
import { POSITION_SCALE, VELOCITY_SCALE } from '../constants/websocket';


/**
 * Visualization system composable that handles:
 * - Three.js scene management
 * - GPU-accelerated rendering
 * - Binary position updates from WebSocket
 * - Fisheye distortion effect
 * - Performance optimizations for large graphs
 */
export function useVisualization() {
  // Store for visualization settings
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
    // Track if we're receiving GPU updates
    isGPUMode: false,
    // Performance metrics
    fps: 0,
    lastFrameTime: 0
  });

  // Efficient storage for node meshes using Map
  const nodeMeshes = new Map<string, THREE.InstancedMesh>();
  
  // Track animation frame for cleanup
  let animationFrameId: number | null = null;

  // Computed settings from store
  const settings = computed(() => settingsStore.getVisualizationSettings);

  /**
   * Main animation loop
   * Handles:
   * - Scene rendering
   * - Controls updates
   * - FPS calculation
   * - Performance monitoring
   */
  const animate = () => {
    if (!state.value.isInitialized) return;

    const { renderer, scene, camera } = state.value;
    if (renderer && scene && camera) {
      // Update FPS counter
      const currentTime = performance.now();
      const delta = currentTime - state.value.lastFrameTime;
      state.value.fps = 1000 / delta;
      state.value.lastFrameTime = currentTime;

      // Render scene
      renderer.render(scene, camera);
    }

    animationFrameId = requestAnimationFrame(animate);
  };

  /**
   * Initialize Three.js scene with optimizations for large graphs
   * Sets up:
   * - WebGL2 renderer with optimizations
   * - Efficient camera and controls
   * - Instanced rendering for nodes
   * - Post-processing effects
   */
  const initScene = () => {
    // Create scene with fog for depth perception
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.Fog(0x000000, 50, 200);

    // Optimized camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 50);

    // Create WebGL2 renderer with optimizations
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      logarithmicDepthBuffer: true // Better z-fighting handling
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.info.autoReset = false; // Manual stats reset for better performance

    // Optimized lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Efficient controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 200;
    controls.minDistance = 10;

    // Store state
    state.value = {
      renderer,
      camera,
      scene,
      canvas: renderer.domElement,
      isInitialized: true,
      isXRSupported: false,
      isWebGL2: renderer.capabilities.isWebGL2,
      isGPUMode: false,
      fps: 0,
      lastFrameTime: performance.now()
    };

    // Start animation loop
    animate();

    return { renderer, scene, camera, controls };
  };

  /**
   * Initialize visualization system
   * Handles:
   * - Scene setup
   * - Window resize
   * - Error handling
   * - WebGL capability detection
   */
  const initialize = async (options: InitializationOptions) => {
    if (state.value.isInitialized) return;

    try {
      console.log('Initializing visualization system...');
      const { renderer, scene, camera } = initScene();

      // Add canvas to container
      const container = document.getElementById('scene-container');
      if (container && renderer.domElement) {
        container.appendChild(renderer.domElement);
      }

      // Efficient resize handler
      const handleResize = () => {
        if (!camera || !renderer) return;
        
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      };
      window.addEventListener('resize', handleResize);

      console.log('Visualization system initialized:', {
        webgl2: state.value.isWebGL2,
        maxTextures: renderer.capabilities.maxTextures,
        precision: renderer.capabilities.precision
      });

    } catch (error) {
      console.error('Failed to initialize visualization:', error);
      throw error;
    }
  };

  /**
   * Handle binary position updates from WebSocket
   * Optimized for:
   * - Efficient updates of large numbers of nodes
   * - Minimal garbage collection
   * - GPU-accelerated rendering
   */
  const updatePositions = (positions: PositionUpdate[], isInitialLayout: boolean) => {
    if (!state.value.scene || !state.value.isInitialized) return;

    // Enable GPU mode on first position update
    if (!state.value.isGPUMode) {
      state.value.isGPUMode = true;
      console.log('Switching to GPU-accelerated mode');
    }

    // Create a map of node IDs to their current meshes
    const meshMap = new Map<string, THREE.InstancedMesh>();
    nodeMeshes.forEach((mesh, id) => {
      meshMap.set(id, mesh);
    });

    // Update node positions efficiently using node IDs
    positions.forEach((pos) => {
      const mesh = meshMap.get(pos.id);
      if (mesh) {
        // Dequantize position values from millimeters to world units
        const x = pos.x / POSITION_SCALE;
        const y = pos.y / POSITION_SCALE;
        const z = pos.z / POSITION_SCALE;
        mesh.position.set(x, y, z);

        // Dequantize velocity values from 0.0001 units to world units
        const vx = pos.vx / VELOCITY_SCALE;
        const vy = pos.vy / VELOCITY_SCALE;
        const vz = pos.vz / VELOCITY_SCALE;
        mesh.userData.velocity = new THREE.Vector3(vx, vy, vz);
      }
    });

    // Force scene update if this is initial layout
    if (isInitialLayout && state.value.camera) {
      // Reset camera to fit all nodes
      const box = new THREE.Box3();
      nodeMeshes.forEach(mesh => box.expandByObject(mesh));
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      state.value.camera.position.copy(center);
      state.value.camera.position.z += maxDim * 2;
      state.value.camera.lookAt(center);
    }
  };

  /**
   * Update node visualization
   * Handles:
   * - Efficient mesh creation/updates
   * - Material updates
   * - Size/color changes
   * - Memory management
   */
  const updateNodes = (nodes: Node[]) => {
    if (!state.value.scene) return;

    // Create shared geometry and material
    const nodeGeometry = new THREE.SphereGeometry(1, 16, 16);
    const nodeMaterial = new THREE.MeshStandardMaterial({
      color: 0x00ff00,
      metalness: 0.3,
      roughness: 0.7,
      transparent: true
    });

    // Update or create meshes efficiently
    nodes.forEach(node => {
      let mesh = nodeMeshes.get(node.id);
      
      if (!mesh) {
        // Create new instanced mesh for better performance
        mesh = new THREE.InstancedMesh(nodeGeometry, nodeMaterial.clone(), 1);
        nodeMeshes.set(node.id, mesh);
        state.value.scene?.add(mesh);
      }

      // Update transform
      if (node.position) {
        mesh.position.set(node.position[0], node.position[1], node.position[2]);
      }

      // Update appearance
      if (node.color && mesh.material instanceof THREE.MeshStandardMaterial) {
        mesh.material.color.setStyle(node.color);
      }

      // Update size
      const scale = node.size || 1;
      mesh.scale.set(scale, scale, scale);
    });

    // Clean up removed nodes
    nodeMeshes.forEach((mesh, id) => {
      if (!nodes.find(node => node.id === id)) {
        state.value.scene?.remove(mesh);
        mesh.geometry.dispose();
        if (mesh.material instanceof THREE.Material) {
          mesh.material.dispose();
        }
        nodeMeshes.delete(id);
      }
    });
  };

  /**
   * Apply fisheye distortion effect
   * @param enabled Whether the effect is active
   * @param strength Distortion strength
   * @param focusPoint Focus point of the distortion
   * @param radius Radius of effect
   */
  const updateFisheyeEffect = (
    enabled: boolean,
    strength: number,
    focusPoint: [number, number, number],
    radius: number
  ) => {
    if (!state.value.isInitialized) return;

    // Apply fisheye distortion to each node
    nodeMeshes.forEach((mesh) => {
      if (!enabled) {
        // Reset position if effect disabled
        if (mesh.userData.originalPosition) {
          mesh.position.copy(mesh.userData.originalPosition);
        }
        return;
      }

      // Store original position
      if (!mesh.userData.originalPosition) {
        mesh.userData.originalPosition = mesh.position.clone();
      }

      // Calculate distortion
      const focus = new THREE.Vector3(...focusPoint);
      const dir = mesh.position.clone().sub(focus);
      const dist = dir.length();
      
      if (dist < radius) {
        const normDist = dist / radius;
        const scale = 1 + strength * (1 - normDist);
        dir.multiplyScalar(scale);
        mesh.position.copy(focus).add(dir);
      }
    });
  };

  /**
   * Clean up visualization system
   * Handles:
   * - Memory cleanup
   * - WebGL context cleanup
   * - Event listener removal
   */
  const dispose = () => {
    console.log('Disposing visualization system...');
    
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    if (!state.value.isInitialized) return;

    // Clean up meshes
    nodeMeshes.forEach((mesh) => {
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    });
    nodeMeshes.clear();

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
  };

  // Clean up on component unmount
  onBeforeUnmount(() => {
    dispose();
  });

  return {
    state,
    initialize,
    updateNodes,
    updatePositions,
    updateFisheyeEffect,
    dispose
  };
}
