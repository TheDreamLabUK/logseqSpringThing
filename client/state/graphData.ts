/**
 * Graph data management and updates
 */

import { GraphData, Node, Edge, Vector3 } from '../core/types';
import { POSITION_SCALE } from '../core/constants';
import { createLogger, binaryToFloat32Array, float32ArrayToPositions } from '../core/utils';

const logger = createLogger('GraphDataManager');

export class GraphDataManager {
  private static instance: GraphDataManager;
  private nodes: Map<string, Node>;
  private edges: Map<string, Edge>;
  private metadata: Record<string, any>;
  private updateListeners: Set<(data: GraphData) => void>;
  private positionUpdateListeners: Set<(nodePositions: Map<string, Vector3>) => void>;

  private constructor() {
    this.nodes = new Map();
    this.edges = new Map();
    this.metadata = {};
    this.updateListeners = new Set();
    this.positionUpdateListeners = new Set();
  }

  static getInstance(): GraphDataManager {
    if (!GraphDataManager.instance) {
      GraphDataManager.instance = new GraphDataManager();
    }
    return GraphDataManager.instance;
  }

  /**
   * Initialize or update the entire graph data
   */
  updateGraphData(data: GraphData): void {
    // Clear existing data
    this.nodes.clear();
    this.edges.clear();

    // Store nodes in Map for O(1) access
    data.nodes.forEach(node => {
      this.nodes.set(node.id, {
        ...node,
        // Ensure all required properties exist
        position: node.position || { x: 0, y: 0, z: 0 },
        velocity: node.velocity || { x: 0, y: 0, z: 0 },
        mass: node.mass || 1,
        label: node.label || node.id
      });
    });

    // Store edges in Map with composite key
    data.edges.forEach(edge => {
      const edgeId = this.createEdgeId(edge.source, edge.target);
      this.edges.set(edgeId, edge);
    });

    // Update metadata
    this.metadata = data.metadata || {};

    // Notify listeners
    this.notifyUpdateListeners();
    logger.log(`Updated graph data: ${this.nodes.size} nodes, ${this.edges.size} edges`);
  }

  /**
   * Handle binary position updates
   */
  updatePositions(buffer: ArrayBuffer): void {
    try {
      const float32Array = binaryToFloat32Array(buffer);
      const positions = float32ArrayToPositions(float32Array);
      const nodePositions = new Map<string, Vector3>();

      // Update node positions
      let i = 0;
      for (const [id, node] of this.nodes) {
        if (i < positions.length) {
          const position = positions[i];
          node.position = {
            x: position.x * POSITION_SCALE,
            y: position.y * POSITION_SCALE,
            z: position.z * POSITION_SCALE
          };
          nodePositions.set(id, node.position);
          i++;
        }
      }

      // Notify position update listeners
      this.notifyPositionUpdateListeners(nodePositions);
    } catch (error) {
      logger.error('Error processing binary position update:', error);
    }
  }

  /**
   * Add or update a single node
   */
  updateNode(node: Node): void {
    this.nodes.set(node.id, {
      ...node,
      position: node.position || { x: 0, y: 0, z: 0 },
      velocity: node.velocity || { x: 0, y: 0, z: 0 },
      mass: node.mass || 1,
      label: node.label || node.id
    });
    this.notifyUpdateListeners();
  }

  /**
   * Add or update a single edge
   */
  updateEdge(edge: Edge): void {
    const edgeId = this.createEdgeId(edge.source, edge.target);
    this.edges.set(edgeId, edge);
    this.notifyUpdateListeners();
  }

  /**
   * Remove a node and its connected edges
   */
  removeNode(nodeId: string): void {
    this.nodes.delete(nodeId);
    
    // Remove connected edges
    for (const [edgeId, edge] of this.edges) {
      if (edge.source === nodeId || edge.target === nodeId) {
        this.edges.delete(edgeId);
      }
    }
    
    this.notifyUpdateListeners();
  }

  /**
   * Remove an edge
   */
  removeEdge(source: string, target: string): void {
    const edgeId = this.createEdgeId(source, target);
    this.edges.delete(edgeId);
    this.notifyUpdateListeners();
  }

  /**
   * Get the current graph data
   */
  getGraphData(): GraphData {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: Array.from(this.edges.values()),
      metadata: this.metadata
    };
  }

  /**
   * Get a specific node by ID
   */
  getNode(id: string): Node | undefined {
    return this.nodes.get(id);
  }

  /**
   * Get a specific edge by source and target IDs
   */
  getEdge(source: string, target: string): Edge | undefined {
    const edgeId = this.createEdgeId(source, target);
    return this.edges.get(edgeId);
  }

  /**
   * Get all edges connected to a node
   */
  getConnectedEdges(nodeId: string): Edge[] {
    return Array.from(this.edges.values()).filter(
      edge => edge.source === nodeId || edge.target === nodeId
    );
  }

  /**
   * Subscribe to graph data updates
   */
  subscribe(listener: (data: GraphData) => void): () => void {
    this.updateListeners.add(listener);
    return () => {
      this.updateListeners.delete(listener);
    };
  }

  /**
   * Subscribe to position updates only
   */
  subscribeToPositionUpdates(
    listener: (nodePositions: Map<string, Vector3>) => void
  ): () => void {
    this.positionUpdateListeners.add(listener);
    return () => {
      this.positionUpdateListeners.delete(listener);
    };
  }

  /**
   * Clear all graph data
   */
  clear(): void {
    this.nodes.clear();
    this.edges.clear();
    this.metadata = {};
    this.notifyUpdateListeners();
  }

  /**
   * Get graph statistics
   */
  getStats(): {
    nodeCount: number;
    edgeCount: number;
    density: number;
  } {
    const nodeCount = this.nodes.size;
    const edgeCount = this.edges.size;
    const maxEdges = nodeCount * (nodeCount - 1) / 2;
    const density = maxEdges > 0 ? edgeCount / maxEdges : 0;

    return {
      nodeCount,
      edgeCount,
      density
    };
  }

  private createEdgeId(source: string, target: string): string {
    // Create a consistent edge ID regardless of source/target order
    return [source, target].sort().join('_');
  }

  private notifyUpdateListeners(): void {
    const data = this.getGraphData();
    this.updateListeners.forEach(listener => {
      try {
        listener(data);
      } catch (error) {
        logger.error('Error in graph update listener:', error);
      }
    });
  }

  private notifyPositionUpdateListeners(nodePositions: Map<string, Vector3>): void {
    this.positionUpdateListeners.forEach(listener => {
      try {
        listener(nodePositions);
      } catch (error) {
        logger.error('Error in position update listener:', error);
      }
    });
  }
}

// Export a singleton instance
export const graphDataManager = GraphDataManager.getInstance();
