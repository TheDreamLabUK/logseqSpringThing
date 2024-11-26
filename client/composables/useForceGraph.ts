import { ref, onBeforeUnmount } from 'vue';
import * as THREE from 'three';
import { useSettingsStore } from '../stores/settings';
import type { Scene, InstancedMesh, Material } from 'three';
import type { Node, Edge } from '../types/core';

interface NodeInstance {
  id: string;
  index: number;
  x: number;
  y: number;
  z: number;
  metadata?: Record<string, any>;
}

interface LinkInstance {
  source: string;
  target: string;
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

interface ForceGraphResources {
  lod: THREE.LOD;
  nodeInstancedMeshes: NodeInstancedMeshes;
  linkInstancedMesh: THREE.InstancedMesh;
  nodeInstances: Map<string, number>;
  linkInstances: Map<string, number>;
  nodeInstanceCount: number;
  linkInstanceCount: number;
}

export function useForceGraph(scene: Scene) {
  const settingsStore = useSettingsStore();
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
    // Create node geometry with different LOD levels
    const highDetailGeometry = new THREE.SphereGeometry(1, 32, 32);
    const mediumDetailGeometry = new THREE.SphereGeometry(1, 16, 16);
    const lowDetailGeometry = new THREE.SphereGeometry(1, 8, 8);

    const settings = settingsStore.getVisualizationSettings;

    // Create node material
    const nodeMaterial = new THREE.MeshPhysicalMaterial({
      metalness: settings.material.node_material_metalness,
      roughness: settings.material.node_material_roughness,
      transparent: true,
      opacity: settings.material.node_material_opacity,
      envMapIntensity: 1.0,
      clearcoat: settings.material.node_material_clearcoat,
      clearcoatRoughness: settings.material.node_material_clearcoat_roughness
    });

    // Create instanced meshes for each LOD level
    const maxInstances = 10000; // Adjust based on expected graph size
    const nodeInstancedMeshes: NodeInstancedMeshes = {
      high: new THREE.InstancedMesh(highDetailGeometry, nodeMaterial.clone(), maxInstances),
      medium: new THREE.InstancedMesh(mediumDetailGeometry, nodeMaterial.clone(), maxInstances),
      low: new THREE.InstancedMesh(lowDetailGeometry, nodeMaterial.clone(), maxInstances)
    };

    // Create LOD
    const lod = new THREE.LOD();
    lod.addLevel(nodeInstancedMeshes.high, 0);
    lod.addLevel(nodeInstancedMeshes.medium, 10);
    lod.addLevel(nodeInstancedMeshes.low, 20);
    scene.add(lod);

    // Create link geometry
    const linkGeometry = new THREE.CylinderGeometry(0.01, 0.01, 1, 8, 1);
    linkGeometry.rotateX(Math.PI / 2); // Align with Z-axis

    // Create link material
    const linkMaterial = new THREE.MeshBasicMaterial({
      color: settings.edge_color,
      transparent: true,
      opacity: settings.edge_opacity,
      depthWrite: false
    });

    // Create instanced mesh for links
    const linkInstancedMesh = new THREE.InstancedMesh(
      linkGeometry,
      linkMaterial,
      maxInstances * 2 // Links typically more numerous than nodes
    );
    scene.add(linkInstancedMesh);

    resources.value = {
      lod,
      nodeInstancedMeshes,
      linkInstancedMesh,
      nodeInstances: new Map(),
      linkInstances: new Map(),
      nodeInstanceCount: 0,
      linkInstanceCount: 0
    };
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

  const updateNodes = () => {
    const res = resources.value;
    if (!res) return;

    // Reset instance count
    res.nodeInstanceCount = 0;

    // Update node instances
    nodes.value.forEach((node: NodeInstance, index: number) => {
      const size = getNodeSize(node);
      const color = getNodeColor(node);
      const emissiveIntensity = calculateEmissiveIntensity(node);

      // Set transform matrix
      tempMatrix.compose(
        new THREE.Vector3(node.x, node.y, node.z),
        tempQuaternion,
        new THREE.Vector3(size, size, size)
      );

      // Update instances for each LOD level
      (Object.values(res.nodeInstancedMeshes) as THREE.InstancedMesh[]).forEach(instancedMesh => {
        instancedMesh.setMatrixAt(index, tempMatrix);
        instancedMesh.setColorAt(index, color);
        (instancedMesh.material as THREE.MeshPhysicalMaterial).emissiveIntensity = emissiveIntensity;
      });

      res.nodeInstances.set(node.id, index);
      res.nodeInstanceCount = Math.max(res.nodeInstanceCount, index + 1);
    });

    // Update instance meshes
    (Object.values(res.nodeInstancedMeshes) as THREE.InstancedMesh[]).forEach(instancedMesh => {
      instancedMesh.count = res.nodeInstanceCount;
      instancedMesh.instanceMatrix.needsUpdate = true;
      if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
    });
  };

  const updateLinks = () => {
    const res = resources.value;
    if (!res) return;

    // Reset instance count
    res.linkInstanceCount = 0;

    // Update link instances
    links.value.forEach((link: LinkInstance, index: number) => {
      const sourceIndex = res.nodeInstances.get(link.source);
      const targetIndex = res.nodeInstances.get(link.target);

      if (sourceIndex === undefined || targetIndex === undefined) return;

      const sourceNode = nodes.value[sourceIndex];
      const targetNode = nodes.value[targetIndex];

      const sourcePos = new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z);
      const targetPos = new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z);

      // Calculate link transform
      const distance = sourcePos.distanceTo(targetPos);
      tempVector.subVectors(targetPos, sourcePos);
      tempQuaternion.setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        tempVector.normalize()
      );

      tempMatrix.compose(
        sourcePos.lerp(targetPos, 0.5), // Position at midpoint
        tempQuaternion,
        new THREE.Vector3(1, 1, distance)
      );

      // Update link instance
      res.linkInstancedMesh.setMatrixAt(index, tempMatrix);
      
      const weight = link.weight || 1;
      const normalizedWeight = Math.min(weight / 10, 1);
      const settings = settingsStore.getVisualizationSettings;
      tempColor.set(settings.edge_color).multiplyScalar(normalizedWeight);
      res.linkInstancedMesh.setColorAt(index, tempColor);

      res.linkInstances.set(`${link.source}-${link.target}`, index);
      res.linkInstanceCount = Math.max(res.linkInstanceCount, index + 1);
    });

    // Update link instance mesh
    res.linkInstancedMesh.count = res.linkInstanceCount;
    res.linkInstancedMesh.instanceMatrix.needsUpdate = true;
    if (res.linkInstancedMesh.instanceColor) {
      res.linkInstancedMesh.instanceColor.needsUpdate = true;
    }
  };

  const updateGraph = (graphNodes: Node[], graphEdges: Edge[]) => {
    // Convert graph data to internal format
    nodes.value = graphNodes.map(node => ({
      id: node.id,
      index: 0,
      x: node.position?.[0] || 0,
      y: node.position?.[1] || 0,
      z: node.position?.[2] || 0,
      metadata: node.metadata
    }));

    links.value = graphEdges.map(edge => ({
      source: edge.source,
      target: edge.target,
      weight: edge.weight
    }));

    // Update visualization
    updateNodes();
    updateLinks();
  };

  const dispose = () => {
    const res = resources.value;
    if (!res) return;

    // Dispose of node resources
    (Object.values(res.nodeInstancedMeshes) as THREE.InstancedMesh[]).forEach(instancedMesh => {
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

    // Remove from scene
    scene.remove(res.lod);
    scene.remove(res.linkInstancedMesh);

    // Clear collections
    res.nodeInstances.clear();
    res.linkInstances.clear();
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
    dispose
  };
}
