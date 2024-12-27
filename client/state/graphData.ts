import { transformGraphData, Node, Edge, GraphData } from '../core/types';
import { createLogger } from '../core/utils';
import { API_ENDPOINTS } from '../core/constants';

const logger = createLogger('GraphDataManager');

// Constants
const THROTTLE_INTERVAL = 16;  // ~60fps max
const NODE_POSITION_SIZE = 24;  // 6 floats * 4 bytes
const FLOATS_PER_NODE = 6;     // x, y, z, vx, vy, vz

interface WebSocketService {
  send(data: ArrayBuffer): void;
}

// Extend Edge interface to include id
interface EdgeWithId extends Edge {
  id: string;
}

export class GraphDataManager {
  private static instance: GraphDataManager;
  private nodes: Map<string, Node>;
  private edges: Map<string, EdgeWithId>;
  private wsService: WebSocketService;
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

  public async fetchInitialData(): Promise<void> {
    try {
      const response = await fetch(API_ENDPOINTS.GRAPH_DATA);
      if (!response.ok) {
        throw new Error(`Failed to fetch graph data: ${response.statusText}`);
      }

      const data = await response.json();
      this.updateGraphData(data);
      logger.info('Initial graph data loaded');
    } catch (error) {
      logger.error('Failed to fetch initial graph data:', error);
      throw error;
    }
  }

  public async fetchPaginatedData(page: number = 1, pageSize: number = 100): Promise<void> {
    try {
      const response = await fetch(`${API_ENDPOINTS.GRAPH_PAGINATED}?page=${page}&pageSize=${pageSize}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch paginated data: ${response.statusText}`);
      }

      const data = await response.json();
      this.updateGraphData(data);
      logger.info(`Paginated data loaded for page ${page}`);
    } catch (error) {
      logger.error('Failed to fetch paginated data:', error);
      throw error;
    }
  }

  async loadInitialGraphData(): Promise<void> {
    try {
      // Start with first page of paginated data
      const pageSize = 100; // Match server default
      const response = await fetch(`${API_ENDPOINTS.GRAPH_PAGINATED}?page=1&pageSize=${pageSize}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch graph data: ${response.statusText}`);
      }

      const data = await response.json();
      const transformedData = transformGraphData(data);
      
      // Update nodes with positions
      this.nodes = new Map(transformedData.nodes.map((node: Node) => [node.id, node]));
      
      // Update edges with IDs
      const edgesWithIds = transformedData.edges.map((edge: Edge) => ({
        ...edge,
        id: this.createEdgeId(edge.source, edge.target)
      }));
      this.edges = new Map(edgesWithIds.map(edge => [edge.id, edge]));
      
      // Update metadata
      this.metadata = {
        ...transformedData.metadata || {},
        pagination: {
          totalPages: data.totalPages,
          currentPage: data.currentPage,
          totalItems: data.totalItems,
          pageSize: data.pageSize
        }
      };

      // Initialize positions and notify listeners
      this.initializeNodePositions();
      this.notifyUpdateListeners();
      
      // Load remaining pages if any
      if (data.totalPages > 1) {
        await this.loadRemainingPages(data.totalPages, pageSize);
      }
      
      logger.log('Initial graph data loaded successfully');
    } catch (error) {
      logger.error('Error loading initial graph data:', error);
      throw error;
    }
  }

  private async loadRemainingPages(totalPages: number, pageSize: number): Promise<void> {
    try {
      // Load remaining pages in parallel with a reasonable chunk size
      const chunkSize = 5;
      for (let i = 2; i <= totalPages; i += chunkSize) {
        const pagePromises = [];
        for (let j = i; j < Math.min(i + chunkSize, totalPages + 1); j++) {
          pagePromises.push(this.loadPage(j, pageSize));
        }
        await Promise.all(pagePromises);
        // Update listeners after each chunk
        this.notifyUpdateListeners();
      }
    } catch (error) {
      logger.error('Error loading remaining pages:', error);
      throw error;
    }
  }

  private async loadPage(page: number, pageSize: number): Promise<void> {
    try {
      const response = await fetch(`${API_ENDPOINTS.GRAPH_PAGINATED}?page=${page}&pageSize=${pageSize}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch page ${page}: ${response.statusText}`);
      }

      const data = await response.json();
      const transformedData = transformGraphData(data);
      
      // Add new nodes
      transformedData.nodes.forEach((node: Node) => {
        if (!this.nodes.has(node.id)) {
          this.nodes.set(node.id, node);
        }
      });
      
      // Add new edges
      transformedData.edges.forEach((edge: Edge) => {
        const edgeId = this.createEdgeId(edge.source, edge.target);
        if (!this.edges.has(edgeId)) {
          this.edges.set(edgeId, { ...edge, id: edgeId });
        }
      });

      logger.debug(`Loaded page ${page} with ${transformedData.nodes.length} nodes`);
    } catch (error) {
      logger.error(`Error loading page ${page}:`, error);
      throw error;
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

  /**
   * Initialize or update the graph data
   */
  updateGraphData(data: any): void {
    // Update nodes
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

      // Store edges in Map with generated IDs
      if (Array.isArray(data.edges)) {
        data.edges.forEach((edge: Edge) => {
          const edgeId = this.createEdgeId(edge.source, edge.target);
          const edgeWithId: EdgeWithId = {
            ...edge,
            id: edgeId
          };
          this.edges.set(edgeId, edgeWithId);
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
   * Setup binary position updates
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
      edges: Array.from(this.edges.values()) as Edge[],
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
        edges: Array.from(this.edges.values()) as Edge[],
        metadata: { ...this.metadata, binaryUpdatesEnabled: enabled }
      });
    });
  }

  public updateNodePositions(positions: Float32Array): void {
    if (!this.binaryUpdatesEnabled) {
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
