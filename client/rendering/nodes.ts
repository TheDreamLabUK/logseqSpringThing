/**
 * Node and edge rendering using InstancedMesh for both
 */

import * as THREE from 'three';
import { MathUtils } from 'three';
import { Node, Edge, Settings } from '../core/types';
import { SceneManager } from './scene';
import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';
import { graphDataManager } from '../state/graphData';

const logger = createLogger('NodeManager');

// Constants for geometry
const NODE_SEGMENTS = 16;
const EDGE_SEGMENTS = 8;
const NODE_SIZE_MULTIPLIER = 1;

// Binary format constants
const BINARY_VERSION = 1.0;
const FLOATS_PER_NODE = 6;  // x, y, z, vx, vy, vz
const VERSION_OFFSET = 1;    // Skip version float

// Reusable objects for matrix calculations
const matrix = new THREE.Matrix4();
const quaternion = new THREE.Quaternion();
const position = new THREE.Vector3();
const scale = new THREE.Vector3(1, 1, 1);

// Edge calculation vectors (reused for efficiency)
const start = new THREE.Vector3();
const end = new THREE.Vector3();
const direction = new THREE.Vector3();
const center = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);
const tempVector = new THREE.Vector3();
const rotationAxis = new THREE.Vector3(1, 0, 0);

export interface NodeMesh extends THREE.Object3D {
  userData: {
    nodeId: string;
  };
}

export class NodeManager {
  private static instance: NodeManager;
  private sceneManager: SceneManager;

  // Instanced meshes - initialized with dummy values, properly set in constructor
  private nodeInstances!: THREE.InstancedMesh;
  private edgeInstances!: THREE.InstancedMesh;

  // State tracking
  private currentNodes: Node[] = [];
  private currentEdges: Edge[] = [];
  private nodeIndices: Map<string, number> = new Map();
  private highlightedNode: string | null = null;
  
  // Edge update batching
  private dirtyEdges: Set<number> = new Set();
  private batchUpdateTimeout: number | null = null;

  // Settings cache
  private currentSettings: {
    nodes: Settings['nodes'];
    edges: Settings['edges'];
  };

  // Unsubscribe function for position updates
  private unsubscribeFromPositionUpdates: (() => void) | null = null;

  private unsubscribers: (() => void)[] = [];

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    
    // Initialize settings cache
    const initialSettings = settingsManager.getCurrentSettings();
    this.currentSettings = {
      nodes: initialSettings.nodes,
      edges: initialSettings.edges
    };
    
    // Listen for global settings changes
    window.addEventListener('settingsChanged', ((event: CustomEvent) => {
      const newSettings = event.detail as Settings;
      this.onSettingsChanged(newSettings);
    }) as EventListener);
    
    // Initialize with proper geometries
    this.initializeGeometries();
    this.initializeInstances();

    // Subscribe to individual settings changes
    this.setupSettingsSubscriptions();
  }

  private initializeGeometries(): void {
    const nodeGeometry = new THREE.SphereGeometry(
      this.currentSettings.nodes.baseSize * NODE_SIZE_MULTIPLIER, 
      NODE_SEGMENTS, 
      NODE_SEGMENTS
    );
    
    const nodeMaterial = new THREE.MeshPhongMaterial({
      color: new THREE.Color(this.currentSettings.nodes.baseColor),
      shininess: 100,
      specular: new THREE.Color('#FFFFFF'),
      transparent: true,
      opacity: this.currentSettings.nodes.opacity
    });

    const edgeGeometry = new THREE.CylinderGeometry(
      this.currentSettings.edges.baseWidth / 4,
      this.currentSettings.edges.baseWidth / 4,
      1,
      EDGE_SEGMENTS
    );
    edgeGeometry.rotateX(Math.PI / 2);
    
    const edgeMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color(this.currentSettings.edges.color),
      transparent: true,
      opacity: this.currentSettings.edges.opacity,
      depthWrite: false
    });

    this.nodeInstances = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, 10000);
    this.edgeInstances = new THREE.InstancedMesh(edgeGeometry, edgeMaterial, 30000);
  }

  private onSettingsChanged(newSettings: Settings): void {
    // Update local settings cache
    this.currentSettings = {
      nodes: newSettings.nodes,
      edges: newSettings.edges
    };

    // Update materials
    const nodeMaterial = this.nodeInstances.material as THREE.MeshPhongMaterial;
    nodeMaterial.color.set(this.currentSettings.nodes.baseColor);
    nodeMaterial.opacity = this.currentSettings.nodes.opacity;

    const edgeMaterial = this.edgeInstances.material as THREE.MeshBasicMaterial;
    edgeMaterial.color.set(this.currentSettings.edges.color);
    edgeMaterial.opacity = this.currentSettings.edges.opacity;

    // Update geometries if size-related settings changed
    if (this.hasGeometrySettingsChanged(newSettings)) {
      this.updateGeometries();
    }

    // Update instances
    if (this.currentNodes.length > 0) {
      this.updateGraph(this.currentNodes, this.currentEdges);
    }
  }

  private hasGeometrySettingsChanged(newSettings: Settings): boolean {
    return (
      newSettings.nodes.baseSize !== this.currentSettings.nodes.baseSize ||
      newSettings.edges.baseWidth !== this.currentSettings.edges.baseWidth
    );
  }

  private updateGeometries(): void {
    // Update node geometry
    const nodeGeometry = new THREE.SphereGeometry(
      this.currentSettings.nodes.baseSize * NODE_SIZE_MULTIPLIER,
      NODE_SEGMENTS,
      NODE_SEGMENTS
    );
    this.nodeInstances.geometry.dispose();
    this.nodeInstances.geometry = nodeGeometry;

    // Update edge geometry
    const edgeGeometry = new THREE.CylinderGeometry(
      this.currentSettings.edges.baseWidth / 4,
      this.currentSettings.edges.baseWidth / 4,
      1,
      EDGE_SEGMENTS
    );
    edgeGeometry.rotateX(Math.PI / 2);
    this.edgeInstances.geometry.dispose();
    this.edgeInstances.geometry = edgeGeometry;
  }

  private initializeInstances(): void {
    try {
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
    } catch (error) {
      logger.error('Error initializing instances:', error);
      throw error;
    }
  }

  private setupSettingsSubscriptions(): void {
    // Subscribe to settings changes
    const nodeSettings = [
      'baseSize', 'baseColor', 'opacity', 'highlightColor'
    ];

    nodeSettings.forEach(setting => {
      const unsubscribe = settingsManager.subscribe('nodes', setting, (value) => {
        try {
          // Update settings cache
          (this.currentSettings.nodes as any)[setting] = value;
          this.onNodeSettingChanged(setting);
        } catch (error) {
          logger.error(`Error handling node setting change for ${setting}:`, error);
        }
      });
      this.unsubscribers.push(unsubscribe);
    });

    const edgeSettings = [
      'baseWidth', 'color', 'opacity'
    ];

    edgeSettings.forEach(setting => {
      const unsubscribe = settingsManager.subscribe('edges', setting, (value) => {
        try {
          // Update settings cache
          (this.currentSettings.edges as any)[setting] = value;
          this.onEdgeSettingChanged(setting);
        } catch (error) {
          logger.error(`Error handling edge setting change for ${setting}:`, error);
        }
      });
      this.unsubscribers.push(unsubscribe);
    });

    // Subscribe to position updates from graphDataManager
    this.unsubscribeFromPositionUpdates = graphDataManager.subscribeToPositionUpdates(
      (positions: Float32Array) => this.updatePositions(positions)
    );
    
    logger.log('NodeManager initialized with settings:', this.currentSettings);
  }

  private onNodeSettingChanged(setting: string): void {
    try {
      switch (setting) {
        case 'baseSize':
          // Update node geometry with new size
          const nodeGeometry = new THREE.SphereGeometry(this.currentSettings.nodes.baseSize * NODE_SIZE_MULTIPLIER, NODE_SEGMENTS, NODE_SEGMENTS);
          this.nodeInstances.geometry.dispose();
          this.nodeInstances.geometry = nodeGeometry;

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
          break;
        case 'baseColor':
          // Update node material
          const nodeMaterial = this.nodeInstances.material as THREE.MeshPhongMaterial;
          nodeMaterial.color.set(this.currentSettings.nodes.baseColor);
          break;
        case 'opacity':
          // Update node material
          const nodeMaterialOpacity = this.nodeInstances.material as THREE.MeshPhongMaterial;
          nodeMaterialOpacity.opacity = this.currentSettings.nodes.opacity;
          break;
        case 'highlightColor':
          // Update highlighted node color
          if (this.highlightedNode) {
            const index = this.nodeIndices.get(this.highlightedNode);
            if (index !== undefined) {
              const color = new THREE.Color(this.currentSettings.nodes.highlightColor);
              this.nodeInstances.setColorAt(index, color);
            }
          }
          break;
        default:
          logger.log(`Unknown node setting changed: ${setting}`);
      }
    } catch (error) {
      logger.error(`Error applying node setting change for ${setting}:`, error);
    }
  }

  private onEdgeSettingChanged(setting: string): void {
    try {
      switch (setting) {
        case 'baseWidth':
          // Update edge geometry with new width
          const edgeGeometry = new THREE.CylinderGeometry(
            this.currentSettings.edges.baseWidth / 4,
            this.currentSettings.edges.baseWidth / 4,
            1,
            EDGE_SEGMENTS
          );
          edgeGeometry.rotateX(Math.PI / 2);
          this.edgeInstances.geometry.dispose();
          this.edgeInstances.geometry = edgeGeometry;

          // Update all edges to account for new width
          this.currentEdges.forEach((edge, index) => {
            const sourceNode = this.currentNodes.find(n => n.id === edge.source);
            const targetNode = this.currentNodes.find(n => n.id === edge.target);
            if (sourceNode && targetNode) {
              this.updateEdgeInstance(index, sourceNode, targetNode);
            }
          });
          this.edgeInstances.instanceMatrix.needsUpdate = true;
          break;
        case 'color':
          // Update edge material
          const edgeMaterial = this.edgeInstances.material as THREE.MeshBasicMaterial;
          edgeMaterial.color.set(this.currentSettings.edges.color);
          break;
        case 'opacity':
          // Update edge material
          const edgeMaterialOpacity = this.edgeInstances.material as THREE.MeshBasicMaterial;
          edgeMaterialOpacity.opacity = this.currentSettings.edges.opacity;
          break;
        default:
          logger.log(`Unknown edge setting changed: ${setting}`);
      }
    } catch (error) {
      logger.error(`Error applying edge setting change for ${setting}:`, error);
    }
  }

  static getInstance(sceneManager: SceneManager): NodeManager {
    if (!NodeManager.instance) {
      NodeManager.instance = new NodeManager(sceneManager);
    }
    return NodeManager.instance;
  }

  getAllNodeMeshes(): NodeMesh[] {
    return [this.nodeInstances as unknown as NodeMesh];
  }

  getNodePosition(nodeId: string): THREE.Vector3 {
    const node = this.currentNodes.find(n => n.id === nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }
    return new THREE.Vector3(
      node.data.position.x,
      node.data.position.y,
      node.data.position.z
    );
  }

  updateNodePosition(nodeId: string, newPosition: THREE.Vector3): void {
    try {
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
    } catch (error) {
      logger.error('Error updating node position:', error);
      throw error;
    }
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

    try {
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
    } catch (error) {
      logger.error('Error processing batch edge update:', error);
      this.dirtyEdges.clear();
    }
  }

  highlightNode(nodeId: string | null): void {
    try {
      if (this.highlightedNode === nodeId) return;

      const color = new THREE.Color();

      if (this.highlightedNode) {
        const prevIndex = this.nodeIndices.get(this.highlightedNode);
        if (prevIndex !== undefined) {
          const node = this.currentNodes[prevIndex];
          color.set(node?.color || this.currentSettings.nodes.baseColor);
          this.nodeInstances.setColorAt(prevIndex, color);
        }
      }

      if (nodeId) {
        const index = this.nodeIndices.get(nodeId);
        if (index !== undefined) {
          color.set(this.currentSettings.nodes.highlightColor);
          this.nodeInstances.setColorAt(index, color);
        }
      }

      this.highlightedNode = nodeId;
      if (this.nodeInstances.instanceColor) {
        this.nodeInstances.instanceColor.needsUpdate = true;
      }
    } catch (error) {
      logger.error('Error highlighting node:', error);
    }
  }

  updateGraph(nodes: Node[], edges: Edge[]): void {
    try {
      this.currentNodes = nodes;
      this.currentEdges = edges;
      this.nodeIndices.clear();
      this.dirtyEdges.clear();

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
        const color = new THREE.Color(node.color || this.currentSettings.nodes.baseColor);
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
    } catch (error) {
      logger.error('Error updating graph:', error);
      throw error;
    }
  }

  private updateEdgeInstance(index: number, sourceNode: Node, targetNode: Node): void {
    try {
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

      const nodeVisualOffset = this.currentSettings.nodes.baseSize * NODE_SIZE_MULTIPLIER;
      scale.set(nodeVisualOffset, nodeVisualOffset, nodeVisualOffset);

      matrix.compose(position, quaternion, scale);
      this.edgeInstances.setMatrixAt(index, matrix);
    } catch (error) {
      logger.error('Error updating edge instance:', error);
    }
  }

  updatePositions(floatArray: Float32Array): void {
    try {
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

      logger.debug(`Updated positions for ${nodeCount} nodes`);
    } catch (error) {
      logger.error('Error updating positions:', error);
    }
  }

  private processNodeChunk(floatArray: Float32Array, startIndex: number, endIndex: number): void {
    try {
      // Reset quaternion to identity
      quaternion.identity();
      
      for (let i = startIndex; i < endIndex; i++) {
        const baseIndex = VERSION_OFFSET + (i * FLOATS_PER_NODE);
        
        // Extract position
        position.set(
          floatArray[baseIndex],
          floatArray[baseIndex + 1],
          floatArray[baseIndex + 2]
        );

        // Keep uniform scale of 1 for position updates
        scale.set(1, 1, 1);
        
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
    } catch (error) {
      logger.error('Error processing node chunk:', error);
    }
  }

  dispose(): void {
    try {
      if (this.batchUpdateTimeout !== null) {
        clearTimeout(this.batchUpdateTimeout);
      }

      // Unsubscribe from position updates
      if (this.unsubscribeFromPositionUpdates) {
        this.unsubscribeFromPositionUpdates();
        this.unsubscribeFromPositionUpdates = null;
      }

      this.nodeInstances.geometry.dispose();
      (this.nodeInstances.material as THREE.Material).dispose();
      this.sceneManager.remove(this.nodeInstances);

      this.edgeInstances.geometry.dispose();
      (this.edgeInstances.material as THREE.Material).dispose();
      this.sceneManager.remove(this.edgeInstances);

      this.dirtyEdges.clear();
      this.unsubscribers.forEach(unsubscribe => unsubscribe());
      this.unsubscribers = [];
      logger.log('NodeManager disposed');
    } catch (error) {
      logger.error('Error disposing NodeManager:', error);
    }
  }
}
