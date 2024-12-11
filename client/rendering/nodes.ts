/**
 * Node and edge rendering using InstancedMesh
 */

import * as THREE from 'three';
import { Node, Edge, Vector3, VisualizationSettings } from '../core/types';
import { settingsManager } from '../state/settings';
import { graphDataManager } from '../state/graphData';
import { SceneManager } from './scene';
import { createLogger, scaleOps } from '../core/utils';
import { MIN_NODE_SIZE, MAX_NODE_SIZE, SERVER_MIN_SIZE, SERVER_MAX_SIZE } from '../core/constants';

const logger = createLogger('NodeManager');

export interface NodeMesh extends THREE.Object3D {
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
  private nodeMaterial!: THREE.MeshPhysicalMaterial;
  private edgeMaterial!: THREE.LineBasicMaterial;
  private highlightMaterial!: THREE.MeshPhysicalMaterial;
  
  // Instanced meshes
  private nodeInstances!: THREE.InstancedMesh;
  private edges!: THREE.LineSegments;
  
  // Node tracking
  private nodePositions: Map<string, Vector3>;
  private nodeIndices: Map<string, number>;
  private nodeSizes: Map<string, number>;
  private highlightedNode: string | null = null;
  
  // Temporary objects for matrix calculations
  private tempMatrix: THREE.Matrix4;
  private tempColor: THREE.Color;
  private tempScale: THREE.Vector3;

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    this.nodePositions = new Map();
    this.nodeIndices = new Map();
    this.nodeSizes = new Map();
    this.tempMatrix = new THREE.Matrix4();
    this.tempColor = new THREE.Color();
    this.tempScale = new THREE.Vector3();

    this.initializeGeometries();
    this.initializeMaterials();
    this.initializeInstances();
    this.setupEventListeners();

    logger.log('NodeManager initialized');
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

    // Get normalized node size
    const serverSize = this.nodeSizes.get(nodeId) || SERVER_MIN_SIZE;
    const normalizedSize = this.normalizeNodeSize(serverSize);
    this.tempScale.set(normalizedSize, normalizedSize, normalizedSize);

    // Update transform matrix with position and scale
    this.tempMatrix.compose(
      new THREE.Vector3(position.x, position.y, position.z),
      new THREE.Quaternion(),
      this.tempScale
    );
    this.nodeInstances.setMatrixAt(index, this.tempMatrix);

    // Mark instance attributes for update
    this.nodeInstances.instanceMatrix.needsUpdate = true;

    // Update edges connected to this node
    const edges = graphDataManager.getGraphData().edges;
    this.updateEdges(edges);
  }

  private normalizeNodeSize(serverSize: number): number {
    return scaleOps.normalizeNodeSize(
      serverSize,
      SERVER_MIN_SIZE,
      SERVER_MAX_SIZE,
      MIN_NODE_SIZE,
      MAX_NODE_SIZE
    );
  }

  private initializeGeometries(): void {
    this.nodeGeometry = new THREE.SphereGeometry(1, 32, 32);
    this.edgeGeometry = new THREE.BufferGeometry();
    logger.log('Geometries initialized');
  }

  private initializeMaterials(): void {
    const settings = settingsManager.getSettings();
    const materialSettings = settingsManager.getNodeMaterialSettings();

    this.nodeMaterial = new THREE.MeshPhysicalMaterial({
      color: settings.nodeColor,
      opacity: settings.nodeOpacity,
      transparent: settings.nodeOpacity < 1,
      metalness: materialSettings.metalness,
      roughness: materialSettings.roughness,
      emissive: new THREE.Color(settings.nodeColor),
      emissiveIntensity: materialSettings.emissiveIntensity,
      clearcoat: materialSettings.clearcoat,
      clearcoatRoughness: materialSettings.clearcoatRoughness,
      reflectivity: materialSettings.reflectivity,
      envMapIntensity: materialSettings.envMapIntensity,
      side: THREE.DoubleSide,
    });

    this.edgeMaterial = new THREE.LineBasicMaterial({
      color: settings.edgeColor,
      opacity: settings.edgeOpacity,
      transparent: settings.edgeOpacity < 1,
      linewidth: settings.edgeWidth,
    });

    this.highlightMaterial = new THREE.MeshPhysicalMaterial({
      color: settings.nodeHighlightColor,
      opacity: settings.nodeOpacity,
      transparent: settings.nodeOpacity < 1,
      metalness: materialSettings.metalness + 0.1,
      roughness: materialSettings.roughness - 0.1,
      emissive: new THREE.Color(settings.nodeHighlightColor),
      emissiveIntensity: materialSettings.emissiveIntensity * 1.5,
      clearcoat: materialSettings.clearcoat,
      clearcoatRoughness: materialSettings.clearcoatRoughness,
      reflectivity: materialSettings.reflectivity,
      envMapIntensity: materialSettings.envMapIntensity * 1.3,
      side: THREE.DoubleSide,
    });

    logger.log('Materials initialized');
  }

  private initializeInstances(): void {
    this.nodeInstances = new THREE.InstancedMesh(
      this.nodeGeometry,
      this.nodeMaterial,
      1000
    );
    this.nodeInstances.count = 0;
    this.nodeInstances.frustumCulled = false;
    this.nodeInstances.castShadow = true;
    this.nodeInstances.receiveShadow = true;

    this.edges = new THREE.LineSegments(this.edgeGeometry, this.edgeMaterial);
    this.edges.frustumCulled = false;

    this.sceneManager.add(this.nodeInstances);
    this.sceneManager.add(this.edges);

    logger.log('Instances initialized');
  }

  private setupEventListeners(): void {
    graphDataManager.subscribe(data => this.updateGraph(data));
    graphDataManager.subscribeToPositionUpdates(positions => this.updatePositions(positions));
    settingsManager.subscribe(settings => this.updateSettings(settings));
    logger.log('Event listeners setup');
  }

  private updateGraph(data: { nodes: Node[]; edges: Edge[] }): void {
    this.updateNodes(data.nodes);
    this.updateEdges(data.edges);
  }

  private updateNodes(nodes: Node[]): void {
    if (nodes.length > this.nodeInstances.count) {
      const newInstancedMesh = new THREE.InstancedMesh(
        this.nodeGeometry,
        this.nodeMaterial,
        Math.ceil(nodes.length * 1.5)
      );
      newInstancedMesh.castShadow = true;
      newInstancedMesh.receiveShadow = true;
      this.sceneManager.remove(this.nodeInstances);
      this.nodeInstances = newInstancedMesh;
      this.sceneManager.add(this.nodeInstances);
    }

    this.nodeInstances.count = nodes.length;
    this.nodePositions.clear();
    this.nodeIndices.clear();
    this.nodeSizes.clear();

    nodes.forEach((node, index) => {
      this.nodePositions.set(node.id, node.position);
      this.nodeIndices.set(node.id, index);
      this.nodeSizes.set(node.id, node.size || SERVER_MIN_SIZE);

      const normalizedSize = this.normalizeNodeSize(node.size || SERVER_MIN_SIZE);
      this.tempScale.set(normalizedSize, normalizedSize, normalizedSize);

      this.tempMatrix.compose(
        new THREE.Vector3(node.position.x, node.position.y, node.position.z),
        new THREE.Quaternion(),
        this.tempScale
      );
      this.nodeInstances.setMatrixAt(index, this.tempMatrix);

      const color = node.color ? new THREE.Color(node.color) : this.tempColor.set(settingsManager.getSettings().nodeColor);
      this.nodeInstances.setColorAt(index, color);
    });

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
        
        const serverSize = this.nodeSizes.get(nodeId) || SERVER_MIN_SIZE;
        const normalizedSize = this.normalizeNodeSize(serverSize);
        this.tempScale.set(normalizedSize, normalizedSize, normalizedSize);

        this.tempMatrix.compose(
          new THREE.Vector3(position.x, position.y, position.z),
          new THREE.Quaternion(),
          this.tempScale
        );
        this.nodeInstances.setMatrixAt(index, this.tempMatrix);
      }
    });

    const edges = graphDataManager.getGraphData().edges;
    this.updateEdges(edges);

    this.nodeInstances.instanceMatrix.needsUpdate = true;
  }

  private updateSettings(settings: VisualizationSettings): void {
    const materialSettings = settingsManager.getNodeMaterialSettings();

    this.nodeMaterial.color.set(settings.nodeColor);
    this.nodeMaterial.opacity = settings.nodeOpacity;
    this.nodeMaterial.transparent = settings.nodeOpacity < 1;
    this.nodeMaterial.metalness = materialSettings.metalness;
    this.nodeMaterial.roughness = materialSettings.roughness;
    this.nodeMaterial.emissive.set(settings.nodeColor);
    this.nodeMaterial.emissiveIntensity = materialSettings.emissiveIntensity;
    this.nodeMaterial.clearcoat = materialSettings.clearcoat;
    this.nodeMaterial.clearcoatRoughness = materialSettings.clearcoatRoughness;
    this.nodeMaterial.reflectivity = materialSettings.reflectivity;
    this.nodeMaterial.envMapIntensity = materialSettings.envMapIntensity;

    this.edgeMaterial.color.set(settings.edgeColor);
    this.edgeMaterial.opacity = settings.edgeOpacity;
    this.edgeMaterial.transparent = settings.edgeOpacity < 1;
    this.edgeMaterial.linewidth = settings.edgeWidth;

    this.highlightMaterial.color.set(settings.nodeHighlightColor);
    this.highlightMaterial.opacity = settings.nodeOpacity;
    this.highlightMaterial.transparent = settings.nodeOpacity < 1;
  }

  highlightNode(nodeId: string | null): void {
    if (this.highlightedNode === nodeId) return;

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
    this.nodeGeometry.dispose();
    this.edgeGeometry.dispose();
    this.nodeMaterial.dispose();
    this.edgeMaterial.dispose();
    this.highlightMaterial.dispose();
    this.sceneManager.remove(this.nodeInstances);
    this.sceneManager.remove(this.edges);
  }
}
