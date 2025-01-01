import { transformGraphData, Node, Edge, GraphData } from '../core/types';
import { createLogger } from '../core/utils';
import { API_ENDPOINTS } from '../core/constants';

const logger = createLogger('GraphDataManager');

// Constants
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
  private wsService!: WebSocketService;  // Use definite assignment assertion
  private metadata: Record<string, any>;
  private updateListeners: Set<(data: GraphData) => void>;
  private positionUpdateListeners: Set<(positions: Float32Array) => void>;
  private binaryUpdatesEnabled: boolean = false;

  private constructor() {
    this.nodes = new Map();
    this.edges = new Map();
    this.metadata = {};
    this.updateListeners = new Set();
    this.positionUpdateListeners = new Set();
    // Initialize with a no-op websocket service
    this.wsService = {
      send: () => logger.warn('WebSocket service not configured')
    };
    this.enableBinaryUpdates();  // Start binary updates by default
  }

  /**
   * Configure the WebSocket service for binary updates
   */
  public setWebSocketService(service: WebSocketService): void {
    this.wsService = service;
    logger.info('WebSocket service configured');
    if (this.binaryUpdatesEnabled) {
      this.updatePositions(new Float32Array());  // Send initial empty update
    }
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
    } catch (_error) {
      logger.error('Failed to fetch initial graph data:', _error);
      throw _error;
    }
  }

  public async fetchPaginatedData(page: number = 1, pageSize: number = 100): Promise<void> {
    try {
      const response = await fetch(
        `${API_ENDPOINTS.GRAPH_PAGINATED}?page=${page}&pageSize=${pageSize}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch paginated data: ${response.statusText}`);
      }

      const data = await response.json();
      this.updateGraphData(data);
      logger.info(`Paginated data loaded for page ${page}`);
    } catch (_error) {
      logger.error('Failed to fetch paginated data:', _error);
      throw _error;
    }
  }

  async loadInitialGraphData(): Promise<void> {
    try {
      // Try both endpoints
      const endpoints = [
        '/api/graph/paginated',
        '/api/graph/data/paginated'
      ];

      let response = null;
      for (const endpoint of endpoints) {
        try {
          response = await fetch(`${endpoint}?page=1&pageSize=100`);
          if (response.ok) break;
        } catch (_e) {
          continue;
        }
      }

      if (!response || !response.ok) {
        throw new Error('Failed to fetch graph data from any endpoint');
      }

      const data = await response.json();
      const transformedData = transformGraphData(data);
      
      // Update nodes and edges
      this.nodes = new Map(transformedData.nodes.map((node: Node) => [node.id, node]));
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

      // Enable WebSocket updates immediately
      this.enableBinaryUpdates();
      this.setBinaryUpdatesEnabled(true);
      
      // Notify listeners of initial data
      this.notifyUpdateListeners();
      
      // Load remaining pages if any
      if (data.totalPages > 1) {
        await this.loadRemainingPages(data.totalPages, data.pageSize);
      }
      
      logger.info('Initial graph data loaded successfully');
    } catch (_error) {
      logger.error('Failed to fetch graph data:', _error);
      throw new Error('Failed to fetch graph data: ' + _error);
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
    } catch (_error) {
      logger.error('Error loading remaining pages:', _error);
      throw _error;
    }
  }

  private async loadPage(page: number, pageSize: number): Promise<void> {
    try {
      const response = await fetch(
        `${API_ENDPOINTS.GRAPH_PAGINATED}?page=${page}&pageSize=${pageSize}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
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
    } catch (_error) {
      logger.error(`Error loading page ${page}:`, _error);
      throw _error;
    }
  }

  /**
   * Enable binary position updates via WebSocket
   */
  public enableBinaryUpdates(): void {
    try {
      const ws = new WebSocket(`wss://${window.location.host}/wss`);
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => {
        logger.info('WebSocket connection established');
        this.setWebSocketService({
          send: (data: ArrayBuffer) => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(data);
            }
          }
        });
        this.setBinaryUpdatesEnabled(true);
      };
      
      ws.onmessage = (event: MessageEvent) => {
        if (event.data instanceof ArrayBuffer) {
          const positions = new Float32Array(event.data);
          this.updateNodePositions(positions);
          this.notifyPositionUpdateListeners(positions);
        }
      };
      
      ws.onerror = (_error) => {
        logger.error('WebSocket error:', _error);
        this.setBinaryUpdatesEnabled(false);
      };
      
      ws.onclose = () => {
        logger.warn('WebSocket connection closed');
        this.setBinaryUpdatesEnabled(false);
        // Attempt to reconnect after a delay
        setTimeout(() => this.enableBinaryUpdates(), 5000);
      };
      
    } catch (_error) {
      logger.error('Failed to initialize WebSocket:', _error);
      this.setBinaryUpdatesEnabled(false);
    }
  }

  /**
   * Update node positions from binary data
   */
  private updatePositions(positions: Float32Array): void {
    if (!this.binaryUpdatesEnabled) return;
    this.wsService.send(positions.buffer);
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
      } catch (_error) {
        logger.error('Error in graph update listener:', _error);
      }
    });
  }

  private notifyPositionUpdateListeners(positions: Float32Array): void {
    this.positionUpdateListeners.forEach(listener => {
      try {
        listener(positions);
      } catch (_error) {
        logger.error('Error in position update listener:', _error);
      }
    });
  }

  public setBinaryUpdatesEnabled(enabled: boolean): void {
    this.binaryUpdatesEnabled = enabled;
    if (enabled) {
      this.updatePositions(new Float32Array());  // Send initial empty update
    }
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
      } catch (_error) {
        logger.error('Error in position update listener:', _error);
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
