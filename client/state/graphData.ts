/**
 * Graph data management with simplified binary updates
 */

import { GraphData, Node, Edge } from '../core/types';
import { createLogger } from '../core/utils';
import { buildApiUrl } from '../core/api';

const logger = createLogger('GraphDataManager');

// Constants
const THROTTLE_INTERVAL = 16;  // ~60fps
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
  private loadingNodes: boolean = false;
  private currentPage: number = 0;
  private hasMorePages: boolean = true;
  private pageSize: number = 100;

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

  async loadInitialGraphData(): Promise<void> {
    try {
      // Reset state
      this.nodes.clear();
      this.edges.clear();
      this.currentPage = 0;
      this.hasMorePages = true;
      this.loadingNodes = false;

      // First, update the graph data from the backend
      try {
        const updateResponse = await fetch(buildApiUrl('graph/update'), {
          method: 'POST',
        });

        if (!updateResponse.ok) {
          logger.warn(`Graph update returned ${updateResponse.status}, continuing with initial load`);
        } else {
          const updateResult = await updateResponse.json();
          logger.log('Graph update result:', updateResult);
        }
      } catch (updateError) {
        logger.warn('Graph update failed, continuing with initial load:', updateError);
      }

      // Then load the first page
      await this.loadNextPage();
      
      // Notify listeners of initial data
      this.notifyUpdateListeners();

      logger.log('Initial graph data loaded:', {
        nodes: this.nodes.size,
        edges: this.edges.size
      });
    } catch (error) {
      logger.error('Failed to load initial graph data:', error);
      // Don't throw here, allow app to continue with empty graph
      this.notifyUpdateListeners();
    }
  }

  private async loadNextPage(): Promise<void> {
    if (this.loadingNodes || !this.hasMorePages) return;

    try {
      this.loadingNodes = true;
      const response = await fetch(buildApiUrl(`graph/data/paginated?page=${this.currentPage}&pageSize=${this.pageSize}`));
      
      if (!response.ok) {
        throw new Error(`Failed to fetch graph data: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      logger.debug('Received graph data:', {
        nodesCount: data.nodes?.length || 0,
        edgesCount: data.edges?.length || 0,
        totalPages: data.totalPages,
        currentPage: data.currentPage,
        metadata: data.metadata
      });
      
      if (!data.nodes || !Array.isArray(data.nodes)) {
        throw new Error('Invalid graph data: nodes array is missing or invalid');
      }
      
      // Update graph with new nodes and edges
      data.nodes.forEach((node: Node) => this.nodes.set(node.id, node));
      if (data.edges && Array.isArray(data.edges)) {
        data.edges.forEach((edge: Edge) => {
          const edgeId = this.createEdgeId(edge.source, edge.target);
          this.edges.set(edgeId, edge);
        });
      }

      // Update pagination state
      this.currentPage = data.currentPage;
      this.hasMorePages = data.currentPage < data.totalPages;

      // Notify listeners of updated data
      this.notifyUpdateListeners();

      logger.log(`Loaded page ${this.currentPage} of graph data: ${this.nodes.size} nodes, ${this.edges.size} edges`);
    } catch (error) {
      logger.error('Failed to load graph data:', error);
      this.hasMorePages = false;  // Stop trying to load more pages on error
    } finally {
      this.loadingNodes = false;
    }
  }

  /**
   * Initialize or update the graph data
   */
  updateGraphData(data: any): void {
    // Update nodes
    if (data.nodes && Array.isArray(data.nodes)) {
      data.nodes.forEach((node: Node) => {
        // Preserve existing position if available, otherwise use server position or generate random
        const existingNode = this.nodes.get(node.id);
        if (!existingNode) {
          // If server didn't provide position, generate random position
          if (!node.data?.position) {
            node.data = node.data || {};
            node.data.position = {
              x: (Math.random() - 0.5) * 100,  // Increased spread
              y: (Math.random() - 0.5) * 100,
              z: (Math.random() - 0.5) * 100
            };
          }
          // Initialize velocity if not present
          if (!node.data.velocity) {
            node.data.velocity = { x: 0, y: 0, z: 0 };
          }
        }
        this.nodes.set(node.id, node);
      });
    }

    // Update edges
    if (data.edges && Array.isArray(data.edges)) {
      data.edges.forEach((edge: Edge) => {
        const edgeId = this.createEdgeId(edge.source, edge.target);
        this.edges.set(edgeId, edge);
      });
    }

    // Update metadata
    if (data.metadata) {
      this.metadata = { ...this.metadata, ...data.metadata };
    }

    // Enable binary updates if we have nodes and it's not already enabled
    if (this.nodes.size > 0 && !this.binaryUpdatesEnabled) {
      this.setupBinaryUpdates();
    }

    // Notify listeners of updates
    this.notifyUpdateListeners();
  }

  /**
   * Setup binary position updates
   */
  private setupBinaryUpdates(): void {
    this.binaryUpdatesEnabled = true;
    // Initialize positions for existing nodes if needed
    this.nodes.forEach(node => {
      if (!node.data?.position) {
        node.data = node.data || {};
        node.data.position = {
          x: (Math.random() - 0.5) * 100,
          y: (Math.random() - 0.5) * 100,
          z: (Math.random() - 0.5) * 100
        };
      }
      if (!node.data.velocity) {
        node.data.velocity = { x: 0, y: 0, z: 0 };
      }
    });
    logger.log('Binary updates enabled');
  }

  public async loadMoreIfNeeded(): Promise<void> {
    if (this.hasMorePages && !this.loadingNodes) {
      await this.loadNextPage();
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

  public setBinaryUpdatesEnabled(enabled: boolean): void {
    this.binaryUpdatesEnabled = enabled;
    logger.info(`Binary updates ${enabled ? 'enabled' : 'disabled'}`);
    
    // Notify listeners of state change
    this.updateListeners.forEach(listener => {
      listener({
        nodes: Array.from(this.nodes.values()),
        edges: Array.from(this.edges.values()),
        metadata: { ...this.metadata, binaryUpdatesEnabled: enabled }
      });
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
