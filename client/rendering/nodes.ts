/**
 * Node and edge rendering using InstancedMesh for both
 */

import {
  Vector3,
  Matrix4,
  Quaternion,
  Color,
  SphereGeometry,
  CylinderGeometry,
  MeshPhongMaterial,
  MeshBasicMaterial,
  Material,
  InstancedMesh,
  Object3D,
  MathUtils
} from 'three';
import { Node, Edge } from '../core/types';
import { SceneManager } from './scene';
import { createLogger } from '../core/utils';
import { settingsManager } from '../state/settings';

const logger = createLogger('NodeManager');

// Constants for geometry
const NODE_SEGMENTS = 16;
const EDGE_SEGMENTS = 8;
const NODE_SIZE_MULTIPLIER = 1; // Increased from 1 to 5 for better visibility

// Binary format constants
const BINARY_VERSION = 1.0;
const FLOATS_PER_NODE = 6;  // x, y, z, vx, vy, vz
const VERSION_OFFSET = 1;    // Skip version float

// Reusable objects for matrix calculations
const matrix = new Matrix4();
const quaternion = new Quaternion();
const position = new Vector3();
const scale = new Vector3(1, 1, 1);

// Edge calculation vectors (reused for efficiency)
const start = new Vector3();
const end = new Vector3();
const direction = new Vector3();
const center = new Vector3();
const UP = new Vector3(0, 1, 0);
const tempVector = new Vector3();
const rotationAxis = new Vector3(1, 0, 0);

export interface NodeMesh extends Object3D {
  userData: {
    nodeId: string;
  };
}

export class NodeManager {
  private static instance: NodeManager;
  private sceneManager: SceneManager;

  // Instanced meshes - initialized with dummy values, properly set in constructor
  private nodeInstances: InstancedMesh;
  private edgeInstances: InstancedMesh;

  // State tracking
  private currentNodes: Node[] = [];
  private currentEdges: Edge[] = [];
  private nodeIndices: Map<string, number> = new Map();
  private highlightedNode: string | null = null;
  
  // Edge update batching
  private dirtyEdges: Set<number> = new Set();
  private batchUpdateTimeout: number | null = null;

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    
    // Get initial settings
    const threeSettings = settingsManager.getThreeJSSettings();
    
    // Initialize with proper geometries
    const nodeGeometry = new SphereGeometry(threeSettings.nodes.size * NODE_SIZE_MULTIPLIER, NODE_SEGMENTS, NODE_SEGMENTS);
    const nodeMaterial = new MeshPhongMaterial({
      color: new Color(threeSettings.nodes.color),
      shininess: 100,
      specular: new Color('#FFFFFF'),
      transparent: true,
      opacity: threeSettings.nodes.opacity
    });

    const edgeGeometry = new CylinderGeometry(
      threeSettings.edges.width / 4,
      threeSettings.edges.width / 4,
      1,
      EDGE_SEGMENTS
    );
    edgeGeometry.rotateX(Math.PI / 2);
    
    const edgeMaterial = new MeshBasicMaterial({
      color: new Color(threeSettings.edges.color),
      transparent: true,
      opacity: threeSettings.edges.opacity,
      depthWrite: false
    });

    this.nodeInstances = new InstancedMesh(nodeGeometry, nodeMaterial, 10000);
    this.edgeInstances = new InstancedMesh(edgeGeometry, edgeMaterial, 30000);
    
    this.initializeInstances();

    // Subscribe to settings changes
    settingsManager.subscribe(() => this.onSettingsChanged());
    
    logger.log('NodeManager initialized with settings:', threeSettings);
  }

  private onSettingsChanged(): void {
    const threeSettings = settingsManager.getThreeJSSettings();

    // Update node geometry with new size
    const nodeGeometry = new SphereGeometry(threeSettings.nodes.size * NODE_SIZE_MULTIPLIER, NODE_SEGMENTS, NODE_SEGMENTS);
    this.nodeInstances.geometry.dispose();
    this.nodeInstances.geometry = nodeGeometry;

    // Update node material
    const nodeMaterial = this.nodeInstances.material as MeshPhongMaterial;
    nodeMaterial.color.set(threeSettings.nodes.color);
    nodeMaterial.opacity = threeSettings.nodes.opacity;
    nodeMaterial.shininess = 100;
    nodeMaterial.specular.set('#FFFFFF');

    // Update edge material
    const edgeMaterial = this.edgeInstances.material as MeshBasicMaterial;
    edgeMaterial.color.set(threeSettings.edges.color);
    edgeMaterial.opacity = threeSettings.edges.opacity;

    // Update all node positions to account for new size
    this.currentNodes.forEach((node, index) => {
      position.set(
        node.data.position.x,
        node.data.position.y,
        node.data.position.z
      );
      matrix.compose(position, quaternion, scale);
      this.nodeInstances.setMatrixAt(index, matrix);
    });
    this.nodeInstances.instanceMatrix.needsUpdate = true;

    // Update all edges to account for new node size
    this.currentEdges.forEach((edge, index) => {
      const sourceNode = this.currentNodes.find(n => n.id === edge.source);
      const targetNode = this.currentNodes.find(n => n.id === edge.target);
      if (sourceNode && targetNode) {
        this.updateEdgeInstance(index, sourceNode, targetNode);
      }
    });
    this.edgeInstances.instanceMatrix.needsUpdate = true;

    logger.log('Visual settings updated:', threeSettings);
  }

  static getInstance(sceneManager: SceneManager): NodeManager {
    if (!NodeManager.instance) {
      NodeManager.instance = new NodeManager(sceneManager);
    }
    return NodeManager.instance;
  }

  private initializeInstances(): void {
    // Initialize node instances
    this.nodeInstances.count = 0;
    this.nodeInstances.frustumCulled = false;

    // Initialize edge instances
    this.edgeInstances.count = 0;
    this.edgeInstances.frustumCulled = false;
    this.edgeInstances.renderOrder = 1;

    // Add to scene
    this.sceneManager.add(this.nodeInstances);
    this.sceneManager.add(this.edgeInstances);

    logger.log('Instances initialized');
  }

  getAllNodeMeshes(): NodeMesh[] {
    return [this.nodeInstances as unknown as NodeMesh];
  }

  getNodePosition(nodeId: string): Vector3 {
    const node = this.currentNodes.find(n => n.id === nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }
    return new Vector3(
      node.data.position.x,
      node.data.position.y,
      node.data.position.z
    );
  }

  updateNodePosition(nodeId: string, newPosition: Vector3): void {
    const index = this.nodeIndices.get(nodeId);
    if (index === undefined) {
      throw new Error(`Node ${nodeId} not found`);
    }

    // Update node position in current nodes array
    const node = this.currentNodes[index];
    if (node) {
      node.data.position = { x: newPosition.x, y: newPosition.y, z: newPosition.z };
    }

    // Update instance matrix
    position.copy(newPosition);
    matrix.compose(position, quaternion, scale);
    this.nodeInstances.setMatrixAt(index, matrix);
    this.nodeInstances.instanceMatrix.needsUpdate = true;

    // Mark connected edges for update
    this.currentEdges.forEach((edge, edgeIndex) => {
      if (edge.source === nodeId || edge.target === nodeId) {
        this.dirtyEdges.add(edgeIndex);
      }
    });

    // Schedule batch edge update
    this.scheduleBatchEdgeUpdate();
  }

  private scheduleBatchEdgeUpdate(): void {
    if (this.batchUpdateTimeout !== null) return;

    this.batchUpdateTimeout = window.setTimeout(() => {
      this.processBatchEdgeUpdate();
      this.batchUpdateTimeout = null;
    }, 16); // Batch updates at ~60fps
  }

  private processBatchEdgeUpdate(): void {
    if (this.dirtyEdges.size === 0) return;

    for (const edgeIndex of this.dirtyEdges) {
      const edge = this.currentEdges[edgeIndex];
      const sourceNode = this.currentNodes.find(n => n.id === edge.source);
      const targetNode = this.currentNodes.find(n => n.id === edge.target);

      if (sourceNode && targetNode) {
        this.updateEdgeInstance(edgeIndex, sourceNode, targetNode);
      }
    }

    this.dirtyEdges.clear();
    this.edgeInstances.instanceMatrix.needsUpdate = true;
  }

  highlightNode(nodeId: string | null): void {
    if (this.highlightedNode === nodeId) return;

    const color = new Color();
    const threeSettings = settingsManager.getThreeJSSettings();

    if (this.highlightedNode) {
      const prevIndex = this.nodeIndices.get(this.highlightedNode);
      if (prevIndex !== undefined) {
        const node = this.currentNodes[prevIndex];
        color.set(node?.color || threeSettings.nodes.color);
        this.nodeInstances.setColorAt(prevIndex, color);
      }
    }

    if (nodeId) {
      const index = this.nodeIndices.get(nodeId);
      if (index !== undefined) {
        color.set(threeSettings.nodes.highlightColor);
        this.nodeInstances.setColorAt(index, color);
      }
    }

    this.highlightedNode = nodeId;
    if (this.nodeInstances.instanceColor) {
      this.nodeInstances.instanceColor.needsUpdate = true;
    }
  }

  updateGraph(nodes: Node[], edges: Edge[]): void {
    this.currentNodes = nodes;
    this.currentEdges = edges;
    this.nodeIndices.clear();
    this.dirtyEdges.clear();

    // Get current settings
    const threeSettings = settingsManager.getThreeJSSettings();

    // Update node instances count and matrices
    this.nodeInstances.count = nodes.length;
    nodes.forEach((node, index) => {
      this.nodeIndices.set(node.id, index);
      
      position.set(
        node.data.position.x,
        node.data.position.y,
        node.data.position.z
      );

      matrix.compose(position, quaternion, scale);
      this.nodeInstances.setMatrixAt(index, matrix);

      // Set node color based on settings
      const color = new Color(node.color || threeSettings.nodes.color);
      this.nodeInstances.setColorAt(index, color);
    });

    this.nodeInstances.instanceMatrix.needsUpdate = true;
    if (this.nodeInstances.instanceColor) {
      this.nodeInstances.instanceColor.needsUpdate = true;
    }

    // Update edge instances
    this.edgeInstances.count = edges.length;
    edges.forEach((edge, index) => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);

      if (sourceNode && targetNode) {
        this.updateEdgeInstance(index, sourceNode, targetNode);
      }
    });

    this.edgeInstances.instanceMatrix.needsUpdate = true;

    logger.log(`Updated graph: ${nodes.length} nodes, ${edges.length} edges`);
  }

  private updateEdgeInstance(index: number, sourceNode: Node, targetNode: Node): void {
    const sourcePos = sourceNode.data.position;
    const targetPos = targetNode.data.position;

    start.set(sourcePos.x, sourcePos.y, sourcePos.z);
    end.set(targetPos.x, targetPos.y, targetPos.z);

    direction.subVectors(end, start);
    const length = direction.length();
    
    if (length < 0.001) return;

    center.addVectors(start, end).multiplyScalar(0.5);
    position.copy(center);

    direction.normalize();
    const angle = Math.acos(MathUtils.clamp(direction.dot(UP), -1, 1));
    tempVector.crossVectors(UP, direction).normalize();
    
    if (tempVector.lengthSq() < 0.001) {
      quaternion.setFromAxisAngle(rotationAxis, direction.dot(UP) > 0 ? 0 : Math.PI);
    } else {
      quaternion.setFromAxisAngle(tempVector, angle);
    }

    // Get current settings
    const threeSettings = settingsManager.getThreeJSSettings();
    const nodeVisualOffset = threeSettings.nodes.size * NODE_SIZE_MULTIPLIER;

    scale.set(nodeVisualOffset, nodeVisualOffset, nodeVisualOffset);

    matrix.compose(position, quaternion, scale);
    this.edgeInstances.setMatrixAt(index, matrix);
  }

  updatePositions(floatArray: Float32Array): void {
    // Check binary version
    const version = floatArray[0];
    if (version !== BINARY_VERSION) {
      logger.warn(`Received binary data version ${version}, expected ${BINARY_VERSION}`);
      return;
    }

    // Calculate number of nodes from array length
    const nodeCount = Math.floor((floatArray.length - VERSION_OFFSET) / FLOATS_PER_NODE);
    
    if (nodeCount > this.currentNodes.length) {
      logger.warn(`Received more nodes than currently tracked: ${nodeCount} > ${this.currentNodes.length}`);
      return;
    }

    // Process node updates in chunks for better performance
    const CHUNK_SIZE = 1000;
    for (let i = 0; i < nodeCount; i += CHUNK_SIZE) {
      const endIndex = Math.min(i + CHUNK_SIZE, nodeCount);
      this.processNodeChunk(floatArray, i, endIndex);
    }

    // Trigger matrix updates
    this.nodeInstances.instanceMatrix.needsUpdate = true;
    
    // Process batched edge updates
    this.processBatchEdgeUpdate();
  }

  private processNodeChunk(floatArray: Float32Array, startIndex: number, endIndex: number): void {
    // Reset quaternion to identity
    quaternion.identity();
    
    const threeSettings = settingsManager.getThreeJSSettings();
    const nodeSize = threeSettings.nodes.size;
    
    for (let i = startIndex; i < endIndex; i++) {
      const baseIndex = VERSION_OFFSET + (i * FLOATS_PER_NODE);
      
      // Extract position
      position.set(
        floatArray[baseIndex],
        floatArray[baseIndex + 1],
        floatArray[baseIndex + 2]
      );

      // Apply uniform scale
      scale.set(nodeSize, nodeSize, nodeSize);
      
      // Create matrix with uniform scaling
      matrix.compose(position, quaternion, scale);
      this.nodeInstances.setMatrixAt(i, matrix);

      // Update node data
      const node = this.currentNodes[i];
      if (node) {
        node.data.position.x = floatArray[baseIndex];
        node.data.position.y = floatArray[baseIndex + 1];
        node.data.position.z = floatArray[baseIndex + 2];
        
        node.data.velocity.x = floatArray[baseIndex + 3];
        node.data.velocity.y = floatArray[baseIndex + 4];
        node.data.velocity.z = floatArray[baseIndex + 5];

        // Mark connected edges for update
        this.currentEdges.forEach((edge, edgeIndex) => {
          if (edge.source === node.id || edge.target === node.id) {
            this.dirtyEdges.add(edgeIndex);
          }
        });
      }
    }
  }

  dispose(): void {
    if (this.batchUpdateTimeout !== null) {
      clearTimeout(this.batchUpdateTimeout);
    }

    this.nodeInstances.geometry.dispose();
    (this.nodeInstances.material as Material).dispose();
    this.sceneManager.remove(this.nodeInstances);

    this.edgeInstances.geometry.dispose();
    (this.edgeInstances.material as Material).dispose();
    this.sceneManager.remove(this.edgeInstances);

    this.dirtyEdges.clear();
    logger.log('NodeManager disposed');
  }
}
