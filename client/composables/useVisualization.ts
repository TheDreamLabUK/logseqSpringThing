import { ref, computed, onBeforeUnmount, provide, markRaw, shallowRef } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useSettingsStore } from '../stores/settings';
import type { Node, Edge, CoreState, InitializationOptions, GraphNode, GraphEdge } from '../types/core';
import type { PositionUpdate } from '../types/websocket';
import { POSITION_SCALE, VELOCITY_SCALE } from '../constants/websocket';
import { VISUALIZATION_CONSTANTS, LIGHT_SETTINGS } from '../constants/visualization';

// Symbol for providing scene to components
export const SCENE_KEY = Symbol('three-scene');

// Adjusted camera settings for scaled positions
const CAMERA_SETTINGS = {
  fov: 60,
  near: 0.01,
  far: 10000,
  position: new THREE.Vector3(0, 0.5, 2),
  target: new THREE.Vector3(0, 0, 0)
};

// Scene settings
const SCENE_SETTINGS = {
  fogNear: 1,
  fogFar: 5,
  gridSize: 2,
  gridDivisions: 20
};

// Helper function to scale positions
const scalePosition = (pos: [number, number, number]): [number, number, number] => [
  pos[0] / POSITION_SCALE,
  pos[1] / POSITION_SCALE,
  pos[2] / POSITION_SCALE
];

export function useVisualization() {
  const settingsStore = useSettingsStore();
  
  // Core visualization state - using shallowRef for Three.js objects
  const state = shallowRef<CoreState>({
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

  // Track meshes for updates - using Maps to avoid reactivity issues
  const nodeMeshes = new Map<string, THREE.Mesh>();
  const edgeMeshes = new Map<string, THREE.Line>();
  const nodeContainer = markRaw(new THREE.Group());
  const edgeContainer = markRaw(new THREE.Group());
  
  // Track animation frame for cleanup
  let animationFrameId: number | null = null;
  let controls: OrbitControls | null = null;

  // Create or update node mesh
  const createNodeMesh = (node: Node): THREE.Mesh => {
    const geometry = new THREE.SphereGeometry(0.02, 32, 32); // Smaller radius for scaled scene
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
      const scaledPos = scalePosition(node.position);
      mesh.position.set(...scaledPos);
    }
    
    // Set scale based on node size
    const size = (node.size || 1) * 0.02; // Smaller scale for scaled scene
    mesh.scale.setScalar(size);

    // Store node data in userData
    mesh.userData = {
      id: node.id,
      type: 'node',
      originalData: { ...node }
    };

    return markRaw(mesh);
  };

  // Create or update edge mesh
  const createEdgeMesh = (edge: Edge, startPos: THREE.Vector3, endPos: THREE.Vector3): THREE.Line => {
    const geometry = new THREE.BufferGeometry().setFromPoints([startPos, endPos]);
    const material = new THREE.LineBasicMaterial({
      color: edge.color || 0xffffff,
      linewidth: (edge.width || 1) * 0.02 // Smaller width for scaled scene
    });

    const line = new THREE.Line(geometry, material);
    line.userData = {
      id: edge.id,
      type: 'edge',
      originalData: { ...edge }
    };

    return markRaw(line);
  };

  // Initialize Three.js scene
  const initScene = (canvas: HTMLCanvasElement) => {
    console.debug('Initializing Three.js scene...');

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.Fog(0x000000, SCENE_SETTINGS.fogNear, SCENE_SETTINGS.fogFar);

    // Add containers to scene
    scene.add(nodeContainer);
    scene.add(edgeContainer);

    // Initialize scene userData
    scene.userData = {
      needsRender: true,
      lastUpdate: performance.now(),
      nodeContainer,
      edgeContainer
    };

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      CAMERA_SETTINGS.fov,
      window.innerWidth / window.innerHeight,
      CAMERA_SETTINGS.near,
      CAMERA_SETTINGS.far
    );
    camera.position.copy(CAMERA_SETTINGS.position);
    camera.lookAt(CAMERA_SETTINGS.target);

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
    const ambientLight = new THREE.AmbientLight(
      LIGHT_SETTINGS.ambient.color,
      LIGHT_SETTINGS.ambient.intensity
    );
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(
      LIGHT_SETTINGS.directional.color,
      LIGHT_SETTINGS.directional.intensity
    );
    directionalLight.position.set(...LIGHT_SETTINGS.directional.position);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    
    const hemiLight = new THREE.HemisphereLight(
      LIGHT_SETTINGS.hemisphere.skyColor,
      LIGHT_SETTINGS.hemisphere.groundColor,
      LIGHT_SETTINGS.hemisphere.intensity
    );
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    // Add controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 5; // Adjusted for scaled scene
    controls.minDistance = 0.1; // Adjusted for scaled scene
    controls.maxPolarAngle = Math.PI * 0.8;
    controls.target.copy(CAMERA_SETTINGS.target);

    // Add grid helper for scale reference
    const gridHelper = new THREE.GridHelper(
      SCENE_SETTINGS.gridSize,
      SCENE_SETTINGS.gridDivisions,
      0x444444,
      0x222222
    );
    scene.add(gridHelper);

    // Add axes helper for orientation
    const axesHelper = new THREE.AxesHelper(1); // Smaller for scaled scene
    scene.add(axesHelper);

    // Provide scene to components
    provide(SCENE_KEY, scene);

    // Return raw objects
    return {
      scene: markRaw(scene),
      camera: markRaw(camera),
      renderer: markRaw(renderer)
    };
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
            edges: edgeMeshes.size,
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

      state.value = markRaw({
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
      });

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
        // Positions are already scaled in WebSocket service
        mesh.position.set(pos.x, pos.y, pos.z);

        // Store velocity in userData for potential use in animations
        mesh.userData.velocity = new THREE.Vector3(pos.vx, pos.vy, pos.vz);

        // Update connected edges
        const nodeData = mesh.userData.originalData as GraphNode;
        if (nodeData.edges) {
          nodeData.edges.forEach((edge: GraphEdge) => {
            const edgeId = `${edge.source}-${edge.target}`;
            const edgeMesh = edgeMeshes.get(edgeId);
            if (edgeMesh) {
              const geometry = edgeMesh.geometry as THREE.BufferGeometry;
              const positions = geometry.getAttribute('position');
              
              if (edge.source === pos.id) {
                positions.setXYZ(0, mesh.position.x, mesh.position.y, mesh.position.z);
              } else if (edge.target === pos.id) {
                positions.setXYZ(1, mesh.position.x, mesh.position.y, mesh.position.z);
              }
              positions.needsUpdate = true;
            }
          });
        }
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
          position: [pos.x, pos.y, pos.z],
          velocity: [pos.vx, pos.vy, pos.vz]
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
          const scaledPos = scalePosition(node.position);
          mesh.position.set(...scaledPos);
        }
        if (node.size) {
          const size = node.size * 0.1;
          mesh.scale.setScalar(size);
        }
        if (node.color) {
          (mesh.material as THREE.MeshStandardMaterial).color.set(node.color);
          (mesh.material as THREE.MeshStandardMaterial).emissive.set(node.color);
        }
      }

      // Update edges if node is a GraphNode
      const graphNode = node as GraphNode;
      if (graphNode.edges) {
        graphNode.edges.forEach(edge => {
          const edgeId = `${edge.source}-${edge.target}`;
          let edgeMesh = edgeMeshes.get(edgeId);
          
          if (!edgeMesh) {
            const startNode = nodeMeshes.get(edge.source);
            const endNode = nodeMeshes.get(edge.target);
            if (startNode && endNode) {
              edgeMesh = createEdgeMesh(edge, startNode.position, endNode.position);
              edgeContainer.add(edgeMesh);
              edgeMeshes.set(edgeId, edgeMesh);
            }
          }
        });
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
        edgeCount: edgeMeshes.size,
        sample: nodes[0] ? {
          id: nodes[0].id,
          position: nodes[0].position ? scalePosition(nodes[0].position) : null,
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

    edgeMeshes.forEach(edge => {
      edge.geometry.dispose();
      (edge.material as THREE.Material).dispose();
    });
    edgeMeshes.clear();

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
