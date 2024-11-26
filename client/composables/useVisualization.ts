import { ref, computed, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useSettingsStore } from '../stores/settings';
import type { Node, Edge, CoreState, InitializationOptions } from '../types/core';

export function useVisualization() {
  const settingsStore = useSettingsStore();
  
  // Core state
  const state = ref<CoreState>({
    renderer: null,
    camera: null,
    scene: null,
    canvas: null,
    isInitialized: false,
    isXRSupported: false,
    isWebGL2: false
  });

  // Node meshes map
  const nodeMeshes = new Map<string, THREE.Mesh>();
  
  // Animation frame ID
  let animationFrameId: number | null = null;

  // Settings
  const settings = computed(() => settingsStore.getVisualizationSettings);

  // Animation loop
  const animate = () => {
    if (!state.value.isInitialized) return;

    const { renderer, scene, camera } = state.value;
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }

    animationFrameId = requestAnimationFrame(animate);
  };

  // Initialize Three.js scene
  const initScene = () => {
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 50);

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Create controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Store state
    state.value = {
      renderer,
      camera,
      scene,
      canvas: renderer.domElement,
      isInitialized: true,
      isXRSupported: false,
      isWebGL2: renderer.capabilities.isWebGL2
    };

    // Start animation loop
    animate();

    return { renderer, scene, camera, controls };
  };

  // Platform initialization
  const initialize = async (options: InitializationOptions) => {
    if (state.value.isInitialized) return;

    try {
      console.log('Initializing Three.js scene...');
      const { renderer, scene, camera } = initScene();

      // Add canvas to container
      const container = document.getElementById('scene-container');
      if (container && renderer.domElement) {
        container.appendChild(renderer.domElement);
      }

      // Handle window resize
      const handleResize = () => {
        if (!camera || !renderer) return;
        
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      };
      window.addEventListener('resize', handleResize);

      console.log('Visualization system initialized successfully');

    } catch (error) {
      console.error('Failed to initialize visualization:', error);
      throw error;
    }
  };

  // Update node positions
  const updateNodes = (nodes: Node[]) => {
    if (!state.value.scene) return;

    const nodeGeometry = new THREE.SphereGeometry(1, 16, 16);
    const nodeMaterial = new THREE.MeshStandardMaterial({
      color: 0x00ff00,
      metalness: 0.3,
      roughness: 0.7
    });

    // Update or create meshes for each node
    nodes.forEach(node => {
      let mesh = nodeMeshes.get(node.id);
      
      if (!mesh) {
        // Create new mesh if it doesn't exist
        mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
        nodeMeshes.set(node.id, mesh);
        state.value.scene?.add(mesh);
      }

      // Update position
      if (node.position) {
        mesh.position.set(node.position[0], node.position[1], node.position[2]);
      }

      // Update color if specified
      if (node.color && mesh.material instanceof THREE.MeshStandardMaterial) {
        mesh.material.color.setStyle(node.color);
      }

      // Update size if specified
      const scale = node.size || 1;
      mesh.scale.set(scale, scale, scale);
    });

    // Remove meshes for nodes that no longer exist
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

  // Cleanup
  const dispose = () => {
    console.log('Disposing visualization system...');
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    if (!state.value.isInitialized) return;

    // Dispose of node meshes
    nodeMeshes.forEach((mesh) => {
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    });
    nodeMeshes.clear();

    // Dispose of renderer
    state.value.renderer?.dispose();
    
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
      isWebGL2: false
    };

    console.log('Visualization system disposed');
  };

  onBeforeUnmount(() => {
    dispose();
  });

  return {
    state,
    initialize,
    updateNodes,
    dispose
  };
}
