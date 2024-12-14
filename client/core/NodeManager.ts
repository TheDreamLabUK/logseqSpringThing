import { Node, Vector3 } from './types';
import * as THREE from 'three';
import { SceneManager } from '../rendering/scene';

export class NodeManager {
  private static instance: NodeManager | null = null;
  private nodeMatrices: Map<string, THREE.Matrix4> = new Map();
  private nodes: Map<string, Node> = new Map();
  private tempVector = new THREE.Vector3();
  private sceneManager: SceneManager;

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
  }

  public static getInstance(sceneManager: SceneManager): NodeManager {
    if (!NodeManager.instance) {
      NodeManager.instance = new NodeManager(sceneManager);
    }
    return NodeManager.instance;
  }

  public addNode(node: Node): void {
    this.nodes.set(node.id, node);
    this.nodeMatrices.set(node.id, new THREE.Matrix4());
    this.updateNodePositionInternal(node.id, node.data.position);
  }

  public removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    this.nodeMatrices.delete(nodeId);
  }

  public updateNodePositions(delta: Vector3): void {
    for (const node of this.nodes.values()) {
      const newPosition = {
        x: node.data.position.x + delta.x,
        y: node.data.position.y + delta.y,
        z: node.data.position.z + delta.z
      };
      this.updateNodePositionInternal(node.id, newPosition);
      node.data.position = newPosition;
    }
  }

  private updateNodePositionInternal(nodeId: string, position: Vector3): void {
    const matrix = this.nodeMatrices.get(nodeId);
    if (matrix) {
      this.tempVector.set(position.x, position.y, position.z);
      matrix.identity();
      matrix.elements[12] = this.tempVector.x;
      matrix.elements[13] = this.tempVector.y;
      matrix.elements[14] = this.tempVector.z;
    }
  }

  public getNodeMatrix(nodeId: string): THREE.Matrix4 | undefined {
    return this.nodeMatrices.get(nodeId);
  }

  public getNodes(): Node[] {
    return Array.from(this.nodes.values());
  }

  public dispose(): void {
    this.nodes.clear();
    this.nodeMatrices.clear();
    NodeManager.instance = null;
  }
}
