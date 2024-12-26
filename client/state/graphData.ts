/**
 * Graph data management with simplified binary updates
 */

import { GraphData, Node, Edge } from '../core/types';
import { createLogger } from '../core/utils';

const logger = createLogger('GraphDataManager');

// Constants
const THROTTLE_INTERVAL = 16;  // ~60fps max
const NODE_POSITION_SIZE = 24;  // 6 floats * 4 bytes
const FLOATS_PER_NODE = 6;     // x, y, z, vx, vy, vz

interface WebSocketService {
  send(data: ArrayBuffer): void;
}

export class GraphDataManager {
  private static instance: GraphDataManager;
  private nodes: Map<string, Node>;
  private edges: Map<string, Edge>;
  private wsService: WebSocketService;
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
    // Initialize with a no-op websocket service
    this.wsService = {
      send: () => logger.warn('WebSocket service not configured')
    };
  }

  /**
   * Configure the WebSocket service for binary updates
   */
  public setWebSocketService(service: WebSocketService): void {
    this.wsService = service;
    logger.info('WebSocket service configured');
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

      // Refresh the graph from existing metadata
      try {
        const refreshResponse = await fetch('/api/graph/refresh', {
          method: 'POST',
        });

        if (!refreshResponse.ok) {
          logger.warn(`Graph refresh returned ${refreshResponse.status}, continuing with initial load`);
        } else {
          const refreshResult = await refreshResponse.json();
          logger.log('Graph refresh result:', refreshResult);
        }
      } catch (refreshError) {
        logger.warn('Graph refresh failed, continuing with initial load:', refreshError);
      }

      // Then load the first page
      await this.loadNextPage();
      
      // Start binary updates only after initial data is loaded
      this.setupBinaryUpdates();

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
      const response = await fetch(`/api/graph/data/paginated?page=${this.currentPage}&pageSize=${this.pageSize}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch graph data: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      logger.debug('Received graph data:', {
        nodesCount: data.nodes?.length || 0,
        edgesCount: data.edges?.length || 0,
        totalPages: data.total_pages,
        currentPage: this.currentPage,
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
      this.currentPage++;
      this.hasMorePages = this.currentPage < data.totalPages;

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

  private initializeNodePositions(): void {
    // Initialize node positions if they don't have positions yet
    this.nodes.forEach(node => {
      if (!node.data.position || (Array.isArray(node.data.position) && node.data.position.every(p => p === null))) {
        // Initialize positions in a larger sphere to match spring_length scale
        const radius = 100; // Match default spring_length
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = radius * Math.cbrt(Math.random()); // Cube root for uniform distribution
        
        node.data.position = {
          x: r * Math.sin(phi) * Math.cos(theta),
          y: r * Math.sin(phi) * Math.sin(theta),
          z: r * Math.cos(phi)
        };
      }
      if (!node.data.velocity) {
        node.data.velocity = { x: 0, y: 0, z: 0 };
      }
    });

    // Create initial position buffer
    const buffer = new ArrayBuffer(this.nodes.size * NODE_POSITION_SIZE);
    const positions = new Float32Array(buffer);
    
    let index = 0;
    this.nodes.forEach(node => {
      const pos = node.data.position;
      positions[index * 6] = typeof pos.x === 'number' ? pos.x : 0;
      positions[index * 6 + 1] = typeof pos.y === 'number' ? pos.y : 0;
      positions[index * 6 + 2] = typeof pos.z === 'number' ? pos.z : 0;
      positions[index * 6 + 3] = node.data.velocity.x;
      positions[index * 6 + 4] = node.data.velocity.y;
      positions[index * 6 + 5] = node.data.velocity.z;
      index++;
    });

    // Notify listeners of initial positions
    this.positionUpdateListeners.forEach(listener => {
      listener(positions);
    });
  }

  private setupBinaryUpdates(): void {
    // Always initialize positions
    this.initializeNodePositions();
    
    if (this.binaryUpdatesEnabled) {
      // Send message to server to start receiving binary updates
      if (window.ws && window.ws.readyState === WebSocket.OPEN) {
        window.ws.send(JSON.stringify({ type: 'enableBinaryUpdates' }));
        logger.log('Requested binary updates from server');
      } else {
        logger.warn('WebSocket not ready, cannot enable binary updates');
      }
    }
  }

  public async loadMoreIfNeeded(): Promise<void> {
    if (this.hasMorePages && !this.loadingNodes) {
      await this.loadNextPage();
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
        if (Array.isArray(node.data.position)) {
          position = {
            x: node.data.position[0],
            y: node.data.position[1], 
            z: node.data.position[2]
          };
        } else {
          position = node.data.position || null;
        }

        this.nodes.set(node.id, {
          ...node,
          data: {
            ...node.data,
            position
          }
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

      // Initialize positions for new nodes
      this.initializeNodePositions();

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
      
      // Send current positions to server to initialize GPU layout
      this.sendPositionsToServer();
      
      logger.log('Enabled binary updates and sent initial positions');
    } else {
      logger.warn('WebSocket not ready, cannot enable binary updates');
    }
  }

  private sendPositionsToServer(): void {
    if (!this.nodes || this.nodes.size === 0) return;

    // Allocate buffer for node positions (no header)
    const buffer = new ArrayBuffer(this.nodes.size * NODE_POSITION_SIZE);
    const positions = new Float32Array(buffer);

    // Pack positions into binary format
    let i = 0;
    for (const node of this.nodes.values()) {
      const pos = node.data.position;
      positions[i * 6] = typeof pos.x === 'number' ? pos.x : 0;
      positions[i * 6 + 1] = typeof pos.y === 'number' ? pos.y : 0;
      positions[i * 6 + 2] = typeof pos.z === 'number' ? pos.z : 0;
      positions[i * 6 + 3] = node.data.velocity.x;
      positions[i * 6 + 4] = node.data.velocity.y;
      positions[i * 6 + 5] = node.data.velocity.z;
      i++;
    }

    // Send binary data (no header)
    this.wsService.send(positions.buffer);
  }

  /**
   * Handle binary position updates with throttling
   */
  updatePositions(positions: Float32Array): void {
    const now = performance.now();
    const timeSinceLastUpdate = now - this.lastUpdateTime;

    if (timeSinceLastUpdate < THROTTLE_INTERVAL) {
      return;  // Skip update if too soon
    }

    try {
      // No version check needed
      // Verify data size (no header)
      const expectedSize = Math.floor(positions.length / FLOATS_PER_NODE) * NODE_POSITION_SIZE;
      if (positions.length * 4 !== expectedSize) {
        logger.error(`Invalid binary data length: ${positions.length * 4} bytes (expected ${expectedSize})`);
        return;
      }

      // Check for invalid values
      let hasInvalidValues = false;
      for (let i = 0; i < positions.length; i++) {
        const val = positions[i];
        if (!Number.isFinite(val) || Math.abs(val) > 1000) {
          logger.warn(`Invalid position value at index ${i}: ${val}`);
          hasInvalidValues = true;
          // Replace invalid value with 0
          positions[i] = 0;
        }
      }

      if (hasInvalidValues) {
        logger.error('Received invalid position values from GPU, using fallback positions');
        // Re-initialize positions
        this.initializeNodePositions();
        return;
      }

      // Log a sample of positions for debugging
      const nodeCount = Math.floor(positions.length / 6);
      logger.debug(`Received positions for ${nodeCount} nodes`);
      if (nodeCount > 0) {
        const firstNode = {
          x: positions[0],
          y: positions[1],
          z: positions[2],
          vx: positions[3],
          vy: positions[4],
          vz: positions[5]
        };
        logger.debug('First node position:', firstNode);
      }

      this.notifyPositionUpdateListeners(positions);
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

  public updateNodePositions(positions: Float32Array): void {
    if (!this.binaryUpdatesEnabled || this.loadingNodes) {
      return;
    }
  
    // Log for debugging
    logger.debug('Received binary position update:', positions);
  
    if (positions.length % FLOATS_PER_NODE !== 0) {
      logger.error('Invalid position array length:', positions.length);
      return;
    }
  
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
