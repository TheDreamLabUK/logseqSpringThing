/**
 * Graph data management with simplified binary updates
 */

import { GraphData, Node, Edge } from '../core/types';
import { createLogger } from '../core/utils';

const logger = createLogger('GraphDataManager');

// Constants
const THROTTLE_INTERVAL = 16;  // ~60fps max
const BINARY_VERSION = 1.0;
const NODE_POSITION_SIZE = 24;  // 6 floats * 4 bytes
const BINARY_HEADER_SIZE = 4;   // 1 float * 4 bytes

export class GraphDataManager {
  private static instance: GraphDataManager;
  private nodes: Map<string, Node>;
  private edges: Map<string, Edge>;
  private metadata: Record<string, any>;
  private updateListeners: Set<(data: GraphData) => void>;
  private positionUpdateListeners: Set<(positions: Float32Array) => void>;
  private lastUpdateTime: number;
  private binaryUpdatesEnabled: boolean = false;

  private constructor() {
    this.nodes = new Map();
    this.edges = new Map();
    this.metadata = {};
    this.updateListeners = new Set();
    this.positionUpdateListeners = new Set();
    this.lastUpdateTime = performance.now();
  }

  static getInstance(): GraphDataManager {
    if (!GraphDataManager.instance) {
      GraphDataManager.instance = new GraphDataManager();
    }
    return GraphDataManager.instance;
  }

  /**
   * Load initial graph data from REST endpoint
   */
  async loadGraphData(): Promise<void> {
    try {
      const response = await fetch('/api/graph/data');
      if (!response.ok) {
        throw new Error(`Failed to fetch graph data: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      this.updateGraphData(data);
      logger.log('Initial graph data loaded from REST endpoint');
    } catch (error) {
      logger.error('Failed to load initial graph data:', error);
      throw error;
    }
  }

  /**
   * Initialize or update the graph data
   */
  updateGraphData(data: any): void {
    logger.log('Received graph data update');

    // Clear existing data
    this.nodes.clear();
    this.edges.clear();

    // Store nodes in Map for O(1) access
    if (data.nodes && Array.isArray(data.nodes)) {
      data.nodes.forEach((node: any) => {
        // Convert position array to object if needed
        let position;
        if (Array.isArray(node.position)) {
          position = {
            x: node.position[0] || 0,
            y: node.position[1] || 0,
            z: node.position[2] || 0
          };
        } else {
          position = node.position || { x: 0, y: 0, z: 0 };
        }

        logger.log(`Processing node ${node.id} with position:`, position);

        this.nodes.set(node.id, {
          ...node,
          position,
          label: node.label || node.id
        });
      });

      // Store edges in Map
      if (Array.isArray(data.edges)) {
        data.edges.forEach((edge: Edge) => {
          const edgeId = this.createEdgeId(edge.source, edge.target);
          this.edges.set(edgeId, edge);
        });
      }

      // Update metadata
      this.metadata = data.metadata || {};

      // Notify listeners
      this.notifyUpdateListeners();
      logger.log(`Updated graph data: ${this.nodes.size} nodes, ${this.edges.size} edges`);

      // Enable binary updates after initial data is received
      if (!this.binaryUpdatesEnabled) {
        this.enableBinaryUpdates();
      }
    } else {
      logger.warn('Invalid graph data format received');
    }
  }

  /**
   * Enable binary position updates
   */
  private enableBinaryUpdates(): void {
    // Send message to server to enable binary updates
    if (window.ws && window.ws.readyState === WebSocket.OPEN) {
      window.ws.send(JSON.stringify({ type: 'enableBinaryUpdates' }));
      this.binaryUpdatesEnabled = true;
      logger.log('Enabled binary updates');
    } else {
      logger.warn('WebSocket not ready, cannot enable binary updates');
    }
  }

  /**
   * Handle binary position updates with throttling
   */
  updatePositions(buffer: ArrayBuffer): void {
    const now = performance.now();
    const timeSinceLastUpdate = now - this.lastUpdateTime;

    if (timeSinceLastUpdate < THROTTLE_INTERVAL) {
      return;  // Skip update if too soon
    }

    try {
      const floatArray = new Float32Array(buffer);
      
      // Check binary version
      const version = floatArray[0];
      if (version !== BINARY_VERSION) {
        logger.warn(`Received binary data version ${version}, expected ${BINARY_VERSION}`);
      }

      // Verify data size
      const expectedSize = BINARY_HEADER_SIZE + Math.floor((buffer.byteLength - BINARY_HEADER_SIZE) / NODE_POSITION_SIZE) * NODE_POSITION_SIZE;
      if (buffer.byteLength !== expectedSize) {
        logger.error(`Invalid binary data length: ${buffer.byteLength} bytes (expected ${expectedSize})`);
        return;
      }

      this.notifyPositionUpdateListeners(floatArray);
      this.lastUpdateTime = now;
    } catch (error) {
      logger.error('Error processing binary position update:', error);
    }
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
    listener: (positions: Float32Array) => void
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

  private createEdgeId(source: string, target: string): string {
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

  private notifyPositionUpdateListeners(positions: Float32Array): void {
    this.positionUpdateListeners.forEach(listener => {
      try {
        listener(positions);
      } catch (error) {
        logger.error('Error in position update listener:', error);
      }
    });
  }
}

// Export a singleton instance
export const graphDataManager = GraphDataManager.getInstance();

// Declare WebSocket on window for TypeScript
declare global {
  interface Window {
    ws: WebSocket;
  }
}
