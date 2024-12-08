import { ref, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { useSettingsStore } from '../stores/settings';
import { useBinaryUpdateStore } from '../stores/binaryUpdate';
import type { Scene, InstancedMesh, Material, Camera, PerspectiveCamera, Vector3 } from 'three';
import type { Node, Edge } from '../types/core';

// Performance tuning constants
const FRAME_SKIP_THRESHOLD = 16; // ms
const FRUSTUM_CULL_MARGIN = 2.0; // Units beyond camera frustum to render
const MAX_VISIBLE_NODES = 10000; // Maximum nodes to render at once
const OBJECT_POOL_SIZE = 1000; // Size of reusable object pool
const MIN_NODE_DISTANCE = 0.1; // Minimum distance between nodes for LOD
const MAX_INSTANCES = 100000; // Maximum number of instances
const SPATIAL_GRID_SIZE = 100; // Size of spatial grid for culling

interface NodeInstance {
  id: string;
  index: number;
  x: number;
  y: number;
  z: number;
  visible: boolean;
  lastUpdateFrame: number;
  metadata?: Record<string, any>;
}

interface LinkInstance {
  source: string;
  target: string;
  visible: boolean;
  lastUpdateFrame: number;
  weight?: number;
}

interface NodeColors {
  NEW: THREE.Color;
  RECENT: THREE.Color;
  MEDIUM: THREE.Color;
  OLD: THREE.Color;
  CORE: THREE.Color;
  SECONDARY: THREE.Color;
  DEFAULT: THREE.Color;
}

interface NodeInstancedMeshes {
  high: THREE.InstancedMesh;
  medium: THREE.InstancedMesh;
  low: THREE.InstancedMesh;
}

interface SpatialGridCell {
  nodes: NodeInstance[];
  center: Vector3;
}

interface ForceGraphResources {
  lod: THREE.LOD;
  nodeInstancedMeshes: NodeInstancedMeshes;
  linkInstancedMesh: THREE.InstancedMesh;
  nodeInstances: Map<string, number>;
  linkInstances: Map<string, number>;
  nodeInstanceCount: number;
  linkInstanceCount: number;
  frustum: THREE.Frustum;
  objectPool: {
    vector3: THREE.Vector3[];
    matrix4: THREE.Matrix4[];
    color: THREE.Color[];
    quaternion: THREE.Quaternion[];
  };
  frameCount: number;
  lastUpdateTime: number;
  visibleNodes: Set<number>;
  visibleLinks: Set<string>;
  spatialGrid: Map<string, SpatialGridCell>;
}

// Helper to check if camera is perspective
const isPerspectiveCamera = (camera: Camera): camera is PerspectiveCamera => {
  return (camera as PerspectiveCamera).isPerspectiveCamera;
};

// Create object pool
const createObjectPool = () => ({
  vector3: Array(OBJECT_POOL_SIZE).fill(null).map(() => new THREE.Vector3()),
  matrix4: Array(OBJECT_POOL_SIZE).fill(null).map(() => new THREE.Matrix4()),
  color: Array(OBJECT_POOL_SIZE).fill(null).map(() => new THREE.Color()),
  quaternion: Array(OBJECT_POOL_SIZE).fill(null).map(() => new THREE.Quaternion())
});

// Helper to get grid cell key
const getGridKey = (x: number, y: number, z: number): string => {
  const gridX = Math.floor(x / SPATIAL_GRID_SIZE);
  const gridY = Math.floor(y / SPATIAL_GRID_SIZE);
  const gridZ = Math.floor(z / SPATIAL_GRID_SIZE);
  return `${gridX},${gridY},${gridZ}`;
};

export function useForceGraph(scene: Scene) {
  const settingsStore = useSettingsStore();
  const binaryUpdateStore = useBinaryUpdateStore();
  const resources = ref<ForceGraphResources | null>(null);
  
  // Temporary objects for matrix calculations
  const tempMatrix = new THREE.Matrix4();
  const tempColor = new THREE.Color();
  const tempVector = new THREE.Vector3();
  const tempQuaternion = new THREE.Quaternion();
  const tempScale = new THREE.Vector3();

  // Data
  const nodes = ref<NodeInstance[]>([]);
  const links = ref<LinkInstance[]>([]);

  // Initialize node colors from settings
  const nodeColors: NodeColors = {
    NEW: new THREE.Color(settingsStore.getVisualizationSettings.node_color_new),
    RECENT: new THREE.Color(settingsStore.getVisualizationSettings.node_color_recent),
    MEDIUM: new THREE.Color(settingsStore.getVisualizationSettings.node_color_medium),
    OLD: new THREE.Color(settingsStore.getVisualizationSettings.node_color_old),
    CORE: new THREE.Color(settingsStore.getVisualizationSettings.node_color_core),
    SECONDARY: new THREE.Color(settingsStore.getVisualizationSettings.node_color_secondary),
    DEFAULT: new THREE.Color(settingsStore.getVisualizationSettings.node_color)
  };

  const initInstancedMeshes = () => {
    // Create optimized geometries
    const highDetailGeometry = new THREE.SphereGeometry(1, 32, 32).toNonIndexed();
    const mediumDetailGeometry = new THREE.SphereGeometry(1, 16, 16).toNonIndexed();
    const lowDetailGeometry = new THREE.SphereGeometry(1, 8, 8).toNonIndexed();

    const settings = settingsStore.getVisualizationSettings;

    // Create optimized material
    const nodeMaterial = new THREE.MeshPhysicalMaterial({
      metalness: settings.material.node_material_metalness,
      roughness: settings.material.node_material_roughness,
      transparent: true,
      opacity: settings.material.node_material_opacity,
      envMapIntensity: 1.0,
      clearcoat: settings.material.node_material_clearcoat,
      clearcoatRoughness: settings.material.node_material_clearcoat_roughness,
      side: THREE.FrontSide,
      flatShading: true,
      vertexColors: true
    });

    // Create instanced meshes with optimizations
    const nodeInstancedMeshes: NodeInstancedMeshes = {
      high: new THREE.InstancedMesh(highDetailGeometry, nodeMaterial.clone(), MAX_INSTANCES),
      medium: new THREE.InstancedMesh(mediumDetailGeometry, nodeMaterial.clone(), MAX_INSTANCES),
      low: new THREE.InstancedMesh(lowDetailGeometry, nodeMaterial.clone(), MAX_INSTANCES)
    };

    // Configure meshes for performance
    Object.values(nodeInstancedMeshes).forEach(mesh => {
      mesh.frustumCulled = true;
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
    });

    // Create LOD with optimized distances
    const lod = new THREE.LOD();
    lod.addLevel(nodeInstancedMeshes.high, 0);
    lod.addLevel(nodeInstancedMeshes.medium, 50);
    lod.addLevel(nodeInstancedMeshes.low, 150);
    scene.add(lod);

    // Create optimized link geometry
    const linkGeometry = new THREE.CylinderGeometry(0.01, 0.01, 1, 6, 1).toNonIndexed();
    linkGeometry.rotateX(Math.PI / 2);

    // Create optimized link material
    const linkMaterial = new THREE.MeshBasicMaterial({
      color: settings.edge_color,
      transparent: true,
      opacity: settings.edge_opacity,
      depthWrite: false,
      side: THREE.FrontSide,
      vertexColors: true
    });

    // Create optimized link mesh
    const linkInstancedMesh = new THREE.InstancedMesh(
      linkGeometry,
      linkMaterial,
      MAX_INSTANCES
    );
    linkInstancedMesh.frustumCulled = true;
    linkInstancedMesh.matrixAutoUpdate = false;
    linkInstancedMesh.castShadow = false;
    linkInstancedMesh.receiveShadow = false;
    scene.add(linkInstancedMesh);

    resources.value = {
      lod,
      nodeInstancedMeshes,
      linkInstancedMesh,
      nodeInstances: new Map(),
      linkInstances: new Map(),
      nodeInstanceCount: 0,
      linkInstanceCount: 0,
      frustum: new THREE.Frustum(),
      objectPool: createObjectPool(),
      frameCount: 0,
      lastUpdateTime: 0,
      visibleNodes: new Set(),
      visibleLinks: new Set(),
      spatialGrid: new Map()
    };
  };

  const updateSpatialGrid = () => {
    const res = resources.value;
    if (!res) return;

    // Clear previous grid
    res.spatialGrid.clear();

    // Add nodes to grid
    nodes.value.forEach((node) => {
      const key = getGridKey(node.x, node.y, node.z);
      let cell = res.spatialGrid.get(key);
      
      if (!cell) {
        const centerX = Math.floor(node.x / SPATIAL_GRID_SIZE) * SPATIAL_GRID_SIZE + SPATIAL_GRID_SIZE / 2;
        const centerY = Math.floor(node.y / SPATIAL_GRID_SIZE) * SPATIAL_GRID_SIZE + SPATIAL_GRID_SIZE / 2;
        const centerZ = Math.floor(node.z / SPATIAL_GRID_SIZE) * SPATIAL_GRID_SIZE + SPATIAL_GRID_SIZE / 2;
        
        cell = {
          nodes: [],
          center: new THREE.Vector3(centerX, centerY, centerZ)
        };
        res.spatialGrid.set(key, cell);
      }
      
      cell.nodes.push(node);
    });
  };

  const isNodeVisible = (node: NodeInstance, camera: Camera): boolean => {
    const res = resources.value;
    if (!res) return false;

    // Get vector from pool
    const poolIndex = node.index % OBJECT_POOL_SIZE;
    const nodePos = res.objectPool.vector3[poolIndex].set(node.x, node.y, node.z);

    // Check if node is within frustum plus margin
    const distance = camera.position.distanceTo(nodePos);
    if (isPerspectiveCamera(camera) && distance > camera.far * FRUSTUM_CULL_MARGIN) {
      return false;
    }

    // Check spatial grid for nearby nodes
    const key = getGridKey(node.x, node.y, node.z);
    const cell = res.spatialGrid.get(key);
    if (cell && cell.nodes.length > 1) {
      // If there are other nodes in the same cell, only show the closest one to camera
      const closestNode = cell.nodes.reduce((closest: NodeInstance, current: NodeInstance) => {
        const currentDist = camera.position.distanceTo(new THREE.Vector3(current.x, current.y, current.z));
        const closestDist = camera.position.distanceTo(new THREE.Vector3(closest.x, closest.y, closest.z));
        return currentDist < closestDist ? current : closest;
      });
      if (closestNode !== node) return false;
    }

    return res.frustum.containsPoint(nodePos);
  };

  const getNodeSize = (node: NodeInstance): number => {
    const settings = settingsStore.getVisualizationSettings;
    const baseSize = (node.metadata?.size || 1) * settings.min_node_size;
    const weight = node.metadata?.weight || 1;
    return Math.min(baseSize * Math.sqrt(weight), settings.max_node_size);
  };

  const getNodeColor = (node: NodeInstance): THREE.Color => {
    const type = node.metadata?.type || 'DEFAULT';
    return nodeColors[type as keyof NodeColors] || nodeColors.DEFAULT;
  };

  const calculateEmissiveIntensity = (node: NodeInstance): number => {
    const settings = settingsStore.getVisualizationSettings;
    const lastModified = node.metadata?.github_last_modified || 
                        node.metadata?.last_modified || 
                        new Date().toISOString();
    const now = Date.now();
    const ageInDays = (now - new Date(lastModified).getTime()) / (24 * 60 * 60 * 1000);
    
    const normalizedAge = Math.min(ageInDays / 30, 1);
    return settings.material.node_emissive_max_intensity - 
           (normalizedAge * (settings.material.node_emissive_max_intensity - 
                           settings.material.node_emissive_min_intensity));
  };

  const updateNodes = (camera: Camera) => {
    const res = resources.value;
    if (!res) return;

    const now = performance.now();
    const timeSinceLastUpdate = now - res.lastUpdateTime;

    // Skip frame if too soon
    if (timeSinceLastUpdate < FRAME_SKIP_THRESHOLD) return;

    // Update frustum for culling
    res.frustum.setFromProjectionMatrix(
      tempMatrix.multiplyMatrices(
        camera.projectionMatrix,
        camera.matrixWorldInverse
      )
    );

    // Get changed nodes from binary update store
    const changedNodes = binaryUpdateStore.getChangedNodes;
    
    // Update spatial grid if nodes have changed
    if (changedNodes.size > 0) {
      updateSpatialGrid();
    }

    // Clear previous visible nodes
    res.visibleNodes.clear();

    // Reset visible instance count
    let visibleCount = 0;

    // Update only changed nodes that are visible
    nodes.value.forEach((node: NodeInstance, index: number) => {
      // Skip if node hasn't changed and was recently updated
      if (!changedNodes.has(index) && 
          node.lastUpdateFrame === res.frameCount - 1) return;

      // Update visibility
      node.visible = isNodeVisible(node, camera);
      if (!node.visible) return;

      // Skip if we've hit max visible nodes
      if (visibleCount >= MAX_VISIBLE_NODES) return;

      // Get objects from pool
      const poolIndex = visibleCount % OBJECT_POOL_SIZE;
      const matrix = res.objectPool.matrix4[poolIndex];
      const position = res.objectPool.vector3[poolIndex];
      const quaternion = res.objectPool.quaternion[poolIndex];

      const size = getNodeSize(node);
      const color = getNodeColor(node);
      const emissiveIntensity = calculateEmissiveIntensity(node);

      // Update transform
      position.set(node.x, node.y, node.z);
      matrix.compose(
        position,
        quaternion,
        tempScale.set(size, size, size)
      );

      // Update instances for each LOD level
      Object.values(res.nodeInstancedMeshes).forEach(instancedMesh => {
        instancedMesh.setMatrixAt(visibleCount, matrix);
        instancedMesh.setColorAt(visibleCount, color);
        (instancedMesh.material as THREE.MeshPhysicalMaterial).emissiveIntensity = emissiveIntensity;
      });

      // Update tracking
      res.nodeInstances.set(node.id, visibleCount);
      res.visibleNodes.add(index);
      node.lastUpdateFrame = res.frameCount;
      visibleCount++;
    });

    // Update instance meshes
    Object.values(res.nodeInstancedMeshes).forEach(instancedMesh => {
      instancedMesh.count = visibleCount;
      instancedMesh.instanceMatrix.needsUpdate = true;
      if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
    });

    res.nodeInstanceCount = visibleCount;
    res.lastUpdateTime = now;
    res.frameCount++;
  };

  const updateLinks = (camera: Camera) => {
    const res = resources.value;
    if (!res) return;

    // Skip if no nodes have changed
    const changedNodes = binaryUpdateStore.getChangedNodes;
    if (changedNodes.size === 0) return;

    // Clear previous visible links
    res.visibleLinks.clear();

    let visibleCount = 0;

    // Update only links connected to changed nodes that are visible
    links.value.forEach((link: LinkInstance, index: number) => {
      const sourceIndex = res.nodeInstances.get(link.source);
      const targetIndex = res.nodeInstances.get(link.target);

      if (sourceIndex === undefined || targetIndex === undefined) return;

      // Skip if neither node has changed and link was recently updated
      if (!changedNodes.has(sourceIndex) && 
          !changedNodes.has(targetIndex) &&
          link.lastUpdateFrame === res.frameCount - 1) return;

      // Skip if either node is not visible
      if (!res.visibleNodes.has(sourceIndex) || !res.visibleNodes.has(targetIndex)) return;

      // Skip if we've hit max visible links
      if (visibleCount >= MAX_VISIBLE_NODES) return;

      // Get objects from pool
      const poolIndex = visibleCount % OBJECT_POOL_SIZE;
      const sourcePos = res.objectPool.vector3[poolIndex];
      const targetPos = res.objectPool.vector3[(poolIndex + 1) % OBJECT_POOL_SIZE];
      const matrix = res.objectPool.matrix4[poolIndex];
      const quaternion = res.objectPool.quaternion[poolIndex];

      const sourceNode = nodes.value[sourceIndex];
      const targetNode = nodes.value[targetIndex];

      sourcePos.set(sourceNode.x, sourceNode.y, sourceNode.z);
      targetPos.set(targetNode.x, targetNode.y, targetNode.z);

      // Calculate link transform
      const distance = sourcePos.distanceTo(targetPos);
      tempVector.subVectors(targetPos, sourcePos);
      quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        tempVector.normalize()
      );

      matrix.compose(
        sourcePos.lerp(targetPos, 0.5),
        quaternion,
        new THREE.Vector3(1, 1, distance)
      );

      // Update link instance
      res.linkInstancedMesh.setMatrixAt(visibleCount, matrix);
      
      const weight = link.weight || 1;
      const normalizedWeight = Math.min(weight / 10, 1);
      const settings = settingsStore.getVisualizationSettings;
      res.objectPool.color[poolIndex]
        .set(settings.edge_color)
        .multiplyScalar(normalizedWeight);
      res.linkInstancedMesh.setColorAt(visibleCount, res.objectPool.color[poolIndex]);

      // Update tracking
      res.linkInstances.set(`${link.source}-${link.target}`, visibleCount);
      res.visibleLinks.add(`${link.source}-${link.target}`);
      link.lastUpdateFrame = res.frameCount;
      visibleCount++;
    });

    // Update link instance mesh
    res.linkInstancedMesh.count = visibleCount;
    res.linkInstancedMesh.instanceMatrix.needsUpdate = true;
    if (res.linkInstancedMesh.instanceColor) {
      res.linkInstancedMesh.instanceColor.needsUpdate = true;
    }

    res.linkInstanceCount = visibleCount;
  };

  const updateGraph = (graphNodes: Node[], graphEdges: Edge[]) => {
    // Convert graph data to internal format
    nodes.value = graphNodes.map((node, index) => ({
      id: node.id,
      index,
      x: node.position?.[0] || 0,
      y: node.position?.[1] || 0,
      z: node.position?.[2] || 0,
      visible: true,
      lastUpdateFrame: -1,
      metadata: node.metadata
    }));

    links.value = graphEdges.map(edge => ({
      source: edge.source,
      target: edge.target,
      visible: true,
      lastUpdateFrame: -1,
      weight: edge.weight
    }));

    // Force full update
    const res = resources.value;
    if (res) {
      res.frameCount++;
      res.lastUpdateTime = 0;
      // Update spatial grid
      updateSpatialGrid();
    }
  };

  const dispose = () => {
    const res = resources.value;
    if (!res) return;

    // Dispose of node resources
    Object.values(res.nodeInstancedMeshes).forEach(instancedMesh => {
      instancedMesh.geometry.dispose();
      if (instancedMesh.material instanceof THREE.Material) {
        instancedMesh.material.dispose();
      } else if (Array.isArray(instancedMesh.material)) {
        instancedMesh.material.forEach((mat: Material) => mat.dispose());
      }
    });

    // Dispose of link resources
    res.linkInstancedMesh.geometry.dispose();
    if (res.linkInstancedMesh.material instanceof THREE.Material) {
      res.linkInstancedMesh.material.dispose();
    } else if (Array.isArray(res.linkInstancedMesh.material)) {
      res.linkInstancedMesh.material.forEach((mat: Material) => mat.dispose());
    }

    // Clear spatial grid
    res.spatialGrid.clear();

    // Remove from scene
    scene.remove(res.lod);
    scene.remove(res.linkInstancedMesh);

    // Clear collections
    res.nodeInstances.clear();
    res.linkInstances.clear();
    res.visibleNodes.clear();
    res.visibleLinks.clear();
    nodes.value = [];
    links.value = [];
    resources.value = null;
  };

  // Initialize on creation
  initInstancedMeshes();

  // Clean up on unmount
  onBeforeUnmount(() => {
    dispose();
  });

  return {
    updateGraph,
    updateNodes,
    updateLinks,
    dispose
  };
}
