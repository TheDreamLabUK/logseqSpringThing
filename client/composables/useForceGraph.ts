import { ref, onBeforeUnmount, markRaw } from 'vue';
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
  vector3: Array(OBJECT_POOL_SIZE).fill(null).map(() => markRaw(new THREE.Vector3())),
  matrix4: Array(OBJECT_POOL_SIZE).fill(null).map(() => markRaw(new THREE.Matrix4())),
  color: Array(OBJECT_POOL_SIZE).fill(null).map(() => markRaw(new THREE.Color())),
  quaternion: Array(OBJECT_POOL_SIZE).fill(null).map(() => markRaw(new THREE.Quaternion()))
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
  const tempMatrix = markRaw(new THREE.Matrix4());
  const tempColor = markRaw(new THREE.Color());
  const tempVector = markRaw(new THREE.Vector3());
  const tempQuaternion = markRaw(new THREE.Quaternion());
  const tempScale = markRaw(new THREE.Vector3());

  // Data
  const nodes = ref<NodeInstance[]>([]);
  const links = ref<LinkInstance[]>([]);

  // Initialize node colors from settings
  const nodeColors: NodeColors = {
    NEW: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color_new)),
    RECENT: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color_recent)),
    MEDIUM: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color_medium)),
    OLD: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color_old)),
    CORE: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color_core)),
    SECONDARY: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color_secondary)),
    DEFAULT: markRaw(new THREE.Color(settingsStore.getVisualizationSettings.node_color))
  };

  const initInstancedMeshes = () => {
    // Create optimized geometries
    const highDetailGeometry = markRaw(new THREE.SphereGeometry(1, 32, 32).toNonIndexed());
    const mediumDetailGeometry = markRaw(new THREE.SphereGeometry(1, 16, 16).toNonIndexed());
    const lowDetailGeometry = markRaw(new THREE.SphereGeometry(1, 8, 8).toNonIndexed());

    const settings = settingsStore.getVisualizationSettings;

    // Create optimized material
    const nodeMaterial = markRaw(new THREE.MeshPhysicalMaterial({
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
    }));

    // Create instanced meshes with optimizations
    const nodeInstancedMeshes: NodeInstancedMeshes = {
      high: markRaw(new THREE.InstancedMesh(highDetailGeometry, nodeMaterial.clone(), MAX_INSTANCES)),
      medium: markRaw(new THREE.InstancedMesh(mediumDetailGeometry, nodeMaterial.clone(), MAX_INSTANCES)),
      low: markRaw(new THREE.InstancedMesh(lowDetailGeometry, nodeMaterial.clone(), MAX_INSTANCES))
    };

    // Configure meshes for performance
    Object.values(nodeInstancedMeshes).forEach(mesh => {
      mesh.frustumCulled = true;
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
    });

    // Create LOD with optimized distances
    const lod = markRaw(new THREE.LOD());
    lod.addLevel(nodeInstancedMeshes.high, 0);
    lod.addLevel(nodeInstancedMeshes.medium, 50);
    lod.addLevel(nodeInstancedMeshes.low, 150);
    scene.add(lod);

    // Create optimized link geometry
    const linkGeometry = markRaw(new THREE.CylinderGeometry(0.01, 0.01, 1, 6, 1).toNonIndexed());
    linkGeometry.rotateX(Math.PI / 2);

    // Create optimized link material
    const linkMaterial = markRaw(new THREE.MeshBasicMaterial({
      color: settings.edge_color,
      transparent: true,
      opacity: settings.edge_opacity,
      depthWrite: false,
      side: THREE.FrontSide,
      vertexColors: true
    }));

    // Create optimized link mesh
    const linkInstancedMesh = markRaw(new THREE.InstancedMesh(
      linkGeometry,
      linkMaterial,
      MAX_INSTANCES
    ));
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
      frustum: markRaw(new THREE.Frustum()),
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

    // Add nodes to grid - only track position for frustum culling
    nodes.value.forEach((node) => {
      const key = getGridKey(node.x, node.y, node.z);
      let cell = res.spatialGrid.get(key);
      
      if (!cell) {
        const centerX = Math.floor(node.x / SPATIAL_GRID_SIZE) * SPATIAL_GRID_SIZE + SPATIAL_GRID_SIZE / 2;
        const centerY = Math.floor(node.y / SPATIAL_GRID_SIZE) * SPATIAL_GRID_SIZE + SPATIAL_GRID_SIZE / 2;
        const centerZ = Math.floor(node.z / SPATIAL_GRID_SIZE) * SPATIAL_GRID_SIZE + SPATIAL_GRID_SIZE / 2;
        
        cell = {
          nodes: [],
          center: markRaw(new THREE.Vector3(centerX, centerY, centerZ))
        };
        res.spatialGrid.set(key, cell);
      }
      
      // Only store node reference, no additional processing
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

    // Use spatial grid only for broad-phase frustum culling
    const key = getGridKey(node.x, node.y, node.z);
    const cell = res.spatialGrid.get(key);
    
    // If cell is completely outside frustum, cull all nodes in it
    if (cell && !res.frustum.containsPoint(cell.center)) {
      const cornerDistance = Math.sqrt(3) * SPATIAL_GRID_SIZE / 2;
      // Only cull if cell is completely outside frustum (including margin)
      if (isPerspectiveCamera(camera) && camera.position.distanceTo(cell.center) > camera.far * FRUSTUM_CULL_MARGIN + cornerDistance) {
        return false;
      }
    }

    // Final per-node frustum check
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
    
    // Only update spatial grid if we have changed nodes
    if (changedNodes.size > 0) {
      updateSpatialGrid();
    }

    // Track current visible nodes for efficient updates
    const newVisibleNodes = new Set<number>();
    let visibleCount = 0;

    // First pass: Update changed nodes and their neighbors
    const nodesToUpdate = new Set<number>();
    changedNodes.forEach(index => {
      nodesToUpdate.add(index);
      // Add neighboring nodes that might be affected
      links.value.forEach(link => {
        const sourceIndex = nodes.value.findIndex(n => n.id === link.source);
        const targetIndex = nodes.value.findIndex(n => n.id === link.target);
        if (sourceIndex === index) nodesToUpdate.add(targetIndex);
        if (targetIndex === index) nodesToUpdate.add(sourceIndex);
      });
    });

    // Second pass: Process nodes that need updating
    nodes.value.forEach((node: NodeInstance, index: number) => {
      const needsUpdate = nodesToUpdate.has(index) || 
                         node.lastUpdateFrame !== res.frameCount - 1;
      
      if (needsUpdate) {
        // Update visibility
        node.visible = isNodeVisible(node, camera);
      }

      if (!node.visible) return;

      // Skip if we've hit max visible nodes
      if (visibleCount >= MAX_VISIBLE_NODES) return;

      // Get objects from pool
      const poolIndex = visibleCount % OBJECT_POOL_SIZE;
      const matrix = res.objectPool.matrix4[poolIndex];
      const position = res.objectPool.vector3[poolIndex];
      const quaternion = res.objectPool.quaternion[poolIndex];

      // Only recalculate these if the node needs updating
      if (needsUpdate) {
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

        node.lastUpdateFrame = res.frameCount;
      }

      // Update tracking
      res.nodeInstances.set(node.id, visibleCount);
      newVisibleNodes.add(index);
      visibleCount++;
    });

    // Update instance meshes only if we have changes
    if (nodesToUpdate.size > 0) {
      Object.values(res.nodeInstancedMeshes).forEach(instancedMesh => {
        instancedMesh.count = visibleCount;
        instancedMesh.instanceMatrix.needsUpdate = true;
        if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
      });
    }

    res.visibleNodes = newVisibleNodes;
    res.nodeInstanceCount = visibleCount;
    res.lastUpdateTime = now;
    res.frameCount++;
  };

  const updateLinks = (camera: Camera) => {
    const res = resources.value;
    if (!res) return;

    // Get changed nodes from binary update store
    const changedNodes = binaryUpdateStore.getChangedNodes;
    if (changedNodes.size === 0) return;

    // Track current visible links for efficient updates
    const newVisibleLinks = new Set<string>();
    let visibleCount = 0;

    // Create a map of affected links for efficient lookup
    const affectedLinks = new Map<string, LinkInstance>();
    links.value.forEach((link, index) => {
      const sourceIndex = nodes.value.findIndex(n => n.id === link.source);
      const targetIndex = nodes.value.findIndex(n => n.id === link.target);
      
      if (changedNodes.has(sourceIndex) || changedNodes.has(targetIndex)) {
        affectedLinks.set(`${link.source}-${link.target}`, link);
      }
    });

    // Only process affected links
    affectedLinks.forEach((link, linkId) => {
      const sourceIndex = res.nodeInstances.get(link.source);
      const targetIndex = res.nodeInstances.get(link.target);

      if (sourceIndex === undefined || targetIndex === undefined) return;

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
        markRaw(new THREE.Vector3(0, 0, 1)),
        tempVector.normalize()
      );

      matrix.compose(
        sourcePos.lerp(targetPos, 0.5),
        quaternion,
        markRaw(new THREE.Vector3(1, 1, distance))
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
      res.linkInstances.set(linkId, visibleCount);
      newVisibleLinks.add(linkId);
      link.lastUpdateFrame = res.frameCount;
      visibleCount++;
    });

    // Update link instance mesh only if we have changes
    if (affectedLinks.size > 0) {
      res.linkInstancedMesh.count = visibleCount;
      res.linkInstancedMesh.instanceMatrix.needsUpdate = true;
      if (res.linkInstancedMesh.instanceColor) {
        res.linkInstancedMesh.instanceColor.needsUpdate = true;
      }
    }

    res.visibleLinks = newVisibleLinks;
    res.linkInstanceCount = visibleCount;
  };

  const updateGraph = (graphNodes: Node[], graphEdges: Edge[]) => {
    const res = resources.value;
    if (!res) return;

    // Reuse existing arrays if possible
    const newNodeCount = graphNodes.length;
    const existingNodeCount = nodes.value.length;

    if (newNodeCount <= existingNodeCount) {
      // Update existing nodes
      graphNodes.forEach((node, index) => {
        const existingNode = nodes.value[index];
        existingNode.id = node.id;
        existingNode.x = node.position?.[0] || 0;
        existingNode.y = node.position?.[1] || 0;
        existingNode.z = node.position?.[2] || 0;
        existingNode.metadata = node.metadata;
        existingNode.lastUpdateFrame = -1;
      });
      // Truncate if new count is smaller
      nodes.value.length = newNodeCount;
    } else {
      // Need to create new nodes for the additional ones
      const newNodes = graphNodes.slice(existingNodeCount).map((node, index) => ({
        id: node.id,
        index: existingNodeCount + index,
        x: node.position?.[0] || 0,
        y: node.position?.[1] || 0,
        z: node.position?.[2] || 0,
        visible: true,
        lastUpdateFrame: -1,
        metadata: node.metadata
      }));
      nodes.value.push(...newNodes);
    }

    // Similar approach for links
    const newLinkCount = graphEdges.length;
    const existingLinkCount = links.value.length;

    if (newLinkCount <= existingLinkCount) {
      // Update existing links
      graphEdges.forEach((edge, index) => {
        const existingLink = links.value[index];
        existingLink.source = edge.source;
        existingLink.target = edge.target;
        existingLink.weight = edge.weight;
        existingLink.lastUpdateFrame = -1;
      });
      // Truncate if new count is smaller
      links.value.length = newLinkCount;
    } else {
      // Need to create new links for the additional ones
      const newLinks = graphEdges.slice(existingLinkCount).map(edge => ({
        source: edge.source,
        target: edge.target,
        visible: true,
        lastUpdateFrame: -1,
        weight: edge.weight
      }));
      links.value.push(...newLinks);
    }

    // Force full update
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
