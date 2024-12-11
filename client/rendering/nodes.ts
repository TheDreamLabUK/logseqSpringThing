/**
 * Node and edge rendering using InstancedMesh
 */

import * as THREE from 'three';
import { Node, Edge, Vector3, VisualizationSettings } from '../core/types';
// import { createLogger } from '../core/utils';
import { settingsManager } from '../state/settings';
import { graphDataManager } from '../state/graphData';
import { SceneManager } from './scene';

// Logger will be used for debugging node creation, updates, and performance metrics
// const __logger = createLogger('NodeManager');

interface NodeMesh extends THREE.Object3D {
  userData: {
    nodeId: string;
  };
}

export class NodeManager {
  private static instance: NodeManager;
  private sceneManager: SceneManager;

  // Geometry instances
  private nodeGeometry!: THREE.SphereGeometry;
  private edgeGeometry!: THREE.BufferGeometry;
  
  // Materials
  private nodeMaterial!: THREE.MeshPhongMaterial;
  private edgeMaterial!: THREE.LineBasicMaterial;
  private highlightMaterial!: THREE.MeshPhongMaterial;
  
  // Instanced meshes
  private nodeInstances!: THREE.InstancedMesh;
  private edges!: THREE.LineSegments;
  
  // Node tracking
  private nodePositions: Map<string, Vector3>;
  private nodeIndices: Map<string, number>;
  private highlightedNode: string | null = null;
  
  // Temporary objects for matrix calculations
  private tempMatrix: THREE.Matrix4;
  private tempColor: THREE.Color;

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    this.nodePositions = new Map();
    this.nodeIndices = new Map();
    this.tempMatrix = new THREE.Matrix4();
    this.tempColor = new THREE.Color();

    this.initializeGeometries();
    this.initializeMaterials();
    this.initializeInstances();
    this.setupEventListeners();
  }

  static getInstance(sceneManager: SceneManager): NodeManager {
    if (!NodeManager.instance) {
      NodeManager.instance = new NodeManager(sceneManager);
    }
    return NodeManager.instance;
  }

  /**
   * Get all node meshes for intersection testing
   */
  getAllNodeMeshes(): NodeMesh[] {
    return [this.nodeInstances as unknown as NodeMesh];
  }

  /**
   * Get position of a specific node
   */
  getNodePosition(nodeId: string): Vector3 {
    const position = this.nodePositions.get(nodeId);
    if (!position) {
      throw new Error(`Node ${nodeId} not found`);
    }
    return position;
  }

  /**
   * Update position of a specific node
   */
  updateNodePosition(nodeId: string, position: Vector3): void {
    const index = this.nodeIndices.get(nodeId);
    if (index === undefined) {
      throw new Error(`Node ${nodeId} not found`);
    }

    // Update position in tracking map
    this.nodePositions.set(nodeId, position);

    // Update transform matrix
    this.tempMatrix.makeTranslation(
      position.x,
      position.y,
      position.z
    );
    this.nodeInstances.setMatrixAt(index, this.tempMatrix);

    // Mark instance attributes for update
    this.nodeInstances.instanceMatrix.needsUpdate = true;

    // Update edges connected to this node
    const edges = graphDataManager.getGraphData().edges;
    this.updateEdges(edges);
  }

  private initializeGeometries(): void {
    // Node geometry
    this.nodeGeometry = new THREE.SphereGeometry(1, 16, 16);
    
    // Edge geometry (will be updated with positions)
    this.edgeGeometry = new THREE.BufferGeometry();
  }

  private initializeMaterials(): void {
    const settings = settingsManager.getSettings();

    // Node material
    this.nodeMaterial = new THREE.MeshPhongMaterial({
      color: settings.nodeColor,
      opacity: settings.nodeOpacity,
      transparent: settings.nodeOpacity < 1,
    });

    // Edge material
    this.edgeMaterial = new THREE.LineBasicMaterial({
      color: settings.edgeColor,
      opacity: settings.edgeOpacity,
      transparent: settings.edgeOpacity < 1,
    });

    // Highlight material
    this.highlightMaterial = new THREE.MeshPhongMaterial({
      color: settings.nodeHighlightColor,
      opacity: settings.nodeOpacity,
      transparent: settings.nodeOpacity < 1,
    });
  }

  private initializeInstances(): void {
    // Create instanced mesh for nodes
    this.nodeInstances = new THREE.InstancedMesh(
      this.nodeGeometry,
      this.nodeMaterial,
      1000 // Initial capacity, will be resized as needed
    );
    this.nodeInstances.count = 0;
    this.nodeInstances.frustumCulled = false;

    // Create edges mesh (will be updated with positions)
    this.edges = new THREE.LineSegments(this.edgeGeometry, this.edgeMaterial);
    this.edges.frustumCulled = false;

    // Add to scene
    this.sceneManager.add(this.nodeInstances);
    this.sceneManager.add(this.edges);
  }

  private setupEventListeners(): void {
    // Listen for graph data updates
    graphDataManager.subscribe(data => this.updateGraph(data));
    
    // Listen for position updates
    graphDataManager.subscribeToPositionUpdates(positions => this.updatePositions(positions));
    
    // Listen for settings updates
    settingsManager.subscribe(settings => this.updateSettings(settings));
  }

  private updateGraph(data: { nodes: Node[]; edges: Edge[] }): void {
    // Update node instances
    this.updateNodes(data.nodes);
    
    // Update edges
    this.updateEdges(data.edges);
  }

  private updateNodes(nodes: Node[]): void {
    // Resize instanced mesh if needed
    if (nodes.length > this.nodeInstances.count) {
      const newInstancedMesh = new THREE.InstancedMesh(
        this.nodeGeometry,
        this.nodeMaterial,
        Math.ceil(nodes.length * 1.5) // Add some buffer
      );
      this.sceneManager.remove(this.nodeInstances);
      this.nodeInstances = newInstancedMesh;
      this.sceneManager.add(this.nodeInstances);
    }

    // Update instance count
    this.nodeInstances.count = nodes.length;

    // Clear tracking maps
    this.nodePositions.clear();
    this.nodeIndices.clear();

    // Update positions and colors
    nodes.forEach((node, index) => {
      // Store position and index
      this.nodePositions.set(node.id, node.position);
      this.nodeIndices.set(node.id, index);

      // Update transform matrix
      this.tempMatrix.makeTranslation(
        node.position.x,
        node.position.y,
        node.position.z
      );
      this.nodeInstances.setMatrixAt(index, this.tempMatrix);

      // Update color
      const color = node.color ? new THREE.Color(node.color) : this.tempColor.set(settingsManager.getSettings().nodeColor);
      this.nodeInstances.setColorAt(index, color);
    });

    // Mark instance attributes for update
    this.nodeInstances.instanceMatrix.needsUpdate = true;
    if (this.nodeInstances.instanceColor) {
      this.nodeInstances.instanceColor.needsUpdate = true;
    }
  }

  private updateEdges(edges: Edge[]): void {
    const positions: number[] = [];
    
    edges.forEach(edge => {
      const sourcePos = this.nodePositions.get(edge.source);
      const targetPos = this.nodePositions.get(edge.target);
      
      if (sourcePos && targetPos) {
        positions.push(
          sourcePos.x, sourcePos.y, sourcePos.z,
          targetPos.x, targetPos.y, targetPos.z
        );
      }
    });

    // Update edge geometry
    this.edgeGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3)
    );
    this.edgeGeometry.attributes.position.needsUpdate = true;
  }

  private updatePositions(positions: Map<string, Vector3>): void {
    positions.forEach((position, nodeId) => {
      const index = this.nodeIndices.get(nodeId);
      if (index !== undefined) {
        this.nodePositions.set(nodeId, position);
        
        // Update transform matrix
        this.tempMatrix.makeTranslation(
          position.x,
          position.y,
          position.z
        );
        this.nodeInstances.setMatrixAt(index, this.tempMatrix);
      }
    });

    // Update edges
    const edges = graphDataManager.getGraphData().edges;
    this.updateEdges(edges);

    // Mark instance attributes for update
    this.nodeInstances.instanceMatrix.needsUpdate = true;
  }

  private updateSettings(settings: VisualizationSettings): void {
    // Update materials
    this.nodeMaterial.color.set(settings.nodeColor);
    this.nodeMaterial.opacity = settings.nodeOpacity;
    this.nodeMaterial.transparent = settings.nodeOpacity < 1;

    this.edgeMaterial.color.set(settings.edgeColor);
    this.edgeMaterial.opacity = settings.edgeOpacity;
    this.edgeMaterial.transparent = settings.edgeOpacity < 1;

    this.highlightMaterial.color.set(settings.nodeHighlightColor);
    this.highlightMaterial.opacity = settings.nodeOpacity;
    this.highlightMaterial.transparent = settings.nodeOpacity < 1;

    // Update node sizes
    this.nodeGeometry.dispose();
    this.nodeGeometry = new THREE.SphereGeometry(settings.nodeSize, 16, 16);
    this.nodeInstances.geometry = this.nodeGeometry;
  }

  highlightNode(nodeId: string | null): void {
    if (this.highlightedNode === nodeId) return;

    // Reset previous highlight
    if (this.highlightedNode) {
      const prevIndex = this.nodeIndices.get(this.highlightedNode);
      if (prevIndex !== undefined) {
        const node = graphDataManager.getNode(this.highlightedNode);
        if (node?.color) {
          this.tempColor.set(node.color);
        } else {
          this.tempColor.set(settingsManager.getSettings().nodeColor);
        }
        this.nodeInstances.setColorAt(prevIndex, this.tempColor);
      }
    }

    // Set new highlight
    if (nodeId) {
      const index = this.nodeIndices.get(nodeId);
      if (index !== undefined) {
        this.tempColor.set(settingsManager.getSettings().nodeHighlightColor);
        this.nodeInstances.setColorAt(index, this.tempColor);
      }
    }

    this.highlightedNode = nodeId;
    if (this.nodeInstances.instanceColor) {
      this.nodeInstances.instanceColor.needsUpdate = true;
    }
  }

  dispose(): void {
    // Dispose of geometries
    this.nodeGeometry.dispose();
    this.edgeGeometry.dispose();

    // Dispose of materials
    this.nodeMaterial.dispose();
    this.edgeMaterial.dispose();
    this.highlightMaterial.dispose();

    // Remove meshes from scene
    this.sceneManager.remove(this.nodeInstances);
    this.sceneManager.remove(this.edges);
  }
}
