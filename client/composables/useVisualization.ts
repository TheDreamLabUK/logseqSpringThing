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

  // Track meshes for updates
  const nodeMeshes = new Map<string, THREE.Mesh>();
  const nodeContainer = new THREE.Group();
  
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

    // Add node container to scene
    scene.add(nodeContainer);

    // Initialize scene userData
    scene.userData = {
      needsRender: true,
      lastUpdate: performance.now(),
      nodeContainer
    };

    // Create camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 30, 50);
    camera.lookAt(0, 0, 0);

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

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    // Add controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 200;
    controls.minDistance = 10;
    controls.maxPolarAngle = Math.PI * 0.8;
    controls.target.set(0, 0, 0);

    // Provide scene to components
    provide(SCENE_KEY, scene);
    return { scene, camera, renderer };
  };

  // Create or update node mesh
  const createNodeMesh = (node: Node) => {
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshStandardMaterial({
      color: node.color || 0xffffff,
      metalness: 0.3,
      roughness: 0.7,
      emissive: node.color || 0xffffff,
      emissiveIntensity: 0.2
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    
    // Set initial position
    if (node.position) {
      mesh.position.set(...node.position);
    }
    
    // Set scale based on node size
    const size = node.size || 1;
    mesh.scale.setScalar(size);

    // Store node data in userData
    mesh.userData = {
      id: node.id,
      type: 'node',
      originalData: { ...node }
    };

    return mesh;
  };

  // Animation loop
  const animate = () => {
    if (!state.value.isInitialized) return;

    const { renderer, scene, camera } = state.value;
    if (renderer && scene && camera) {
      controls?.update();

      const currentTime = performance.now();
      const delta = currentTime - state.value.lastFrameTime;
      state.value.fps = 1000 / delta;
      state.value.lastFrameTime = currentTime;

      const needsRender = scene.userData?.needsRender !== false || 
                         controls?.enabled || 
                         currentTime - (scene.userData?.lastUpdate || 0) > 1000;

      if (needsRender) {
        renderer.render(scene, camera);
        scene.userData.needsRender = false;
        scene.userData.lastUpdate = currentTime;

        // Log performance stats in development
        if (process.env.NODE_ENV === 'development' && currentTime % 5000 < 16) {
          console.debug('Render stats:', {
            fps: state.value.fps.toFixed(1),
            meshes: nodeMeshes.size,
            drawCalls: renderer.info.render.calls,
            triangles: renderer.info.render.triangles
          });
        }
      }
    }

    animationFrameId = requestAnimationFrame(animate);
  };

  // Initialize visualization system
  const initialize = async (options: InitializationOptions) => {
    if (state.value.isInitialized) return;

    try {
      const { scene, camera, renderer } = initScene(options.canvas);

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

      animate();

      // Handle window resize
      const handleResize = () => {
        if (!camera || !renderer) return;
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      };
      window.addEventListener('resize', handleResize);

      console.log('Visualization system initialized');
    } catch (error) {
      console.error('Failed to initialize visualization:', error);
      throw error;
    }
  };

  // Handle binary position updates
  const updatePositions = (positions: PositionUpdate[], isInitialLayout: boolean) => {
    if (!state.value.scene || !state.value.isInitialized) return;

    positions.forEach(pos => {
      const mesh = nodeMeshes.get(pos.id);
      if (mesh) {
        // Apply scaling factors to convert from quantized values
        mesh.position.set(
          pos.x / POSITION_SCALE,
          pos.y / POSITION_SCALE,
          pos.z / POSITION_SCALE
        );

        // Store velocity in userData for potential use in animations
        mesh.userData.velocity = new THREE.Vector3(
          pos.vx / VELOCITY_SCALE,
          pos.vy / VELOCITY_SCALE,
          pos.vz / VELOCITY_SCALE
        );
      }
    });

    if (state.value.scene) {
      state.value.scene.userData.needsRender = true;
    }

    // Log update in development
    if (process.env.NODE_ENV === 'development') {
      console.debug('Position update:', {
        count: positions.length,
        isInitial: isInitialLayout,
        meshCount: nodeMeshes.size,
        sample: positions[0] ? {
          id: positions[0].id,
          position: [
            positions[0].x / POSITION_SCALE,
            positions[0].y / POSITION_SCALE,
            positions[0].z / POSITION_SCALE
          ]
        } : null
      });
    }
  };

  // Update nodes
  const updateNodes = (nodes: Node[]) => {
    if (!state.value.scene) return;

    // Remove old nodes
    const currentIds = new Set(nodes.map(n => n.id));
    nodeMeshes.forEach((mesh, id) => {
      if (!currentIds.has(id)) {
        nodeContainer.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
        nodeMeshes.delete(id);
      }
    });

    // Add or update nodes
    nodes.forEach(node => {
      let mesh = nodeMeshes.get(node.id);
      
      if (!mesh) {
        // Create new mesh
        mesh = createNodeMesh(node);
        nodeContainer.add(mesh);
        nodeMeshes.set(node.id, mesh);
      } else {
        // Update existing mesh
        if (node.position) {
          mesh.position.set(...node.position);
        }
        if (node.size) {
          mesh.scale.setScalar(node.size);
        }
        if (node.color) {
          (mesh.material as THREE.MeshStandardMaterial).color.set(node.color);
          (mesh.material as THREE.MeshStandardMaterial).emissive.set(node.color);
        }
      }
    });

    if (state.value.scene) {
      state.value.scene.userData.needsRender = true;
    }

    // Log update in development
    if (process.env.NODE_ENV === 'development') {
      console.debug('Nodes updated:', {
        count: nodes.length,
        meshCount: nodeMeshes.size,
        sample: nodes[0] ? {
          id: nodes[0].id,
          position: nodes[0].position,
          size: nodes[0].size
        } : null
      });
    }
  };

  // Cleanup
  onBeforeUnmount(() => {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
    }

    if (controls) {
      controls.dispose();
    }

    // Clean up meshes
    nodeMeshes.forEach(mesh => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    nodeMeshes.clear();

    if (state.value.renderer) {
      state.value.renderer.dispose();
      state.value.renderer.forceContextLoss();
    }
    
    state.value.canvas?.remove();
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
  });

  return {
    state,
    initialize,
    updateNodes,
    updatePositions
  };
}
