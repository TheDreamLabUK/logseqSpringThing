import { transformGraphData, Node, Edge, GraphData } from '../core/types';
import { createLogger } from '../core/utils';
import { Vector3 } from 'three';
import { API_ENDPOINTS } from '../core/constants';
import { debugState } from '../core/debugState';

import { WebSocketService as WebSocketServiceClass } from '../websocket/websocketService';
const logger = createLogger('GraphDataManager');

// Constants
const FLOATS_PER_NODE = 6;     // x, y, z, vx, vy, vz

// Throttling for debug logs
let lastDebugLogTime = 0;
const DEBUG_LOG_THROTTLE_MS = 2000; // Only log once every 2 seconds

// Helper for throttled debug logging
function throttledDebugLog(message: string, data?: any): void {
  if (!debugState.isDataDebugEnabled()) return;
  
  const now = Date.now();
  if (now - lastDebugLogTime > DEBUG_LOG_THROTTLE_MS) {
    lastDebugLogTime = now;
    logger.debug(message, data);
  }
}

// Interface for the internal WebSocket service used by this class
type InternalWebSocketService = { send(data: ArrayBuffer): void };

// Extend Edge interface to include id
interface EdgeWithId extends Edge {
  id: string;
}

// Update NodePosition type to use THREE.Vector3
type NodePosition = Vector3;

export class GraphDataManager {
  private static instance: GraphDataManager;
  private nodes: Map<string, Node>;
  private edges: Map<string, EdgeWithId>;
  private wsService!: InternalWebSocketService;  // Use definite assignment assertion
  private metadata: Record<string, any>;
  private updateListeners: Set<(data: GraphData) => void>;
  private positionUpdateListeners: Set<(positions: Float32Array) => void>;
  private binaryUpdatesEnabled: boolean = false;
  private positionUpdateBuffer: Map<string, NodePosition> = new Map();
  private updateBufferTimeout: number | null = null;
  private static readonly BUFFER_FLUSH_INTERVAL = 16; // ~60fps

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
    // Don't enable binary updates by default
    this.binaryUpdatesEnabled = false;
  }

  /**
   * Configure the WebSocket service for binary updates
   */
  public setWebSocketService(service: InternalWebSocketService): void {
    this.wsService = service;
    logger.info('WebSocket service configured');
    
    // If binary updates were enabled before the service was configured,
    // send an initial empty update now that we have a service
    if (this.binaryUpdatesEnabled) {
      try {
        this.updatePositions(new Float32Array());
        logger.info('Sent initial empty update after WebSocket service configuration');
      } catch (error) {
        logger.error('Failed to send initial update after WebSocket service configuration:', error);
      }
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
      // Start with first page
      throttledDebugLog('Fetching initial graph data page');
      await this.fetchPaginatedData(1, 100);
      
      throttledDebugLog(`Initial graph data page loaded. Current nodes: ${this.nodes.size}, edges: ${this.edges.size}`);
      
      // Get total pages from metadata
      const totalPages = this.metadata.pagination?.totalPages || 1;
      const totalItems = this.metadata.pagination?.totalItems || 0;
      
      if (totalPages > 1) {
        throttledDebugLog(`Loading remaining ${totalPages - 1} pages in background. Total items: ${totalItems}, Current items: ${this.nodes.size}`);
        // Load remaining pages in background with improved error handling
        this.loadRemainingPagesWithRetry(totalPages, 100);
      }
    } catch (error) {
      logger.error('Failed to fetch initial graph data:', error);
      throw error;
    }
  }

  /**
   * Load remaining pages with retry mechanism
   * This runs in the background and doesn't block the initial rendering
   */
  private async loadRemainingPagesWithRetry(totalPages: number, pageSize: number): Promise<void> {
    // Start from page 2 since page 1 is already loaded
    for (let page = 2; page <= totalPages; page++) {
      let retries = 0;
      const maxRetries = 3;
      let success = false;
      
      while (!success && retries < maxRetries) {
        try {
          await this.fetchPaginatedData(page, pageSize);
          success = true;
          throttledDebugLog(`Loaded page ${page}/${totalPages} successfully`);
        } catch (error) {
          retries++;
          const delay = Math.min(1000 * Math.pow(2, retries), 10000); // Exponential backoff with max 10s
          
          logger.warn(`Failed to load page ${page}/${totalPages}, attempt ${retries}/${maxRetries}. Retrying in ${delay}ms...`);
          
          // Wait before retrying
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
      
      if (!success) {
        logger.error(`Failed to load page ${page}/${totalPages} after ${maxRetries} attempts`);
      }
      
      // Notify listeners after each page, even if it failed
      this.notifyUpdateListeners();
      
      // Small delay between pages to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // All pages are now loaded, enable randomization
    try {
      const websocketService = WebSocketServiceClass.getInstance();
      logger.info(`Finished loading all ${totalPages} pages. Enabling server-side randomization.`);
      websocketService.enableRandomization(true);
    } catch (error) {
      logger.warn('Failed to enable randomization after loading all pages:', error);
    }
    logger.info(`Finished loading all ${totalPages} pages. Total nodes: ${this.nodes.size}, edges: ${this.edges.size}`);
  }

  public async fetchPaginatedData(page: number = 1, pageSize: number = 100): Promise<void> {
    try {
      throttledDebugLog(`Fetching page ${page} with size ${pageSize}. Current nodes: ${this.nodes.size}`);
      
      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(
        `${API_ENDPOINTS.GRAPH_PAGINATED}?page=${page}&pageSize=${pageSize}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          signal: controller.signal
        }
      );
      
      // Clear the timeout
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch paginated data: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      throttledDebugLog(`Received data for page ${page}:`, { nodes: data.nodes?.length, edges: data.edges?.length, totalItems: data.totalItems });
      
      this.updateGraphData(data);
      throttledDebugLog(`Paginated data loaded for page ${page}. Total nodes now: ${this.nodes.size}, edges: ${this.edges.size}`);
    } catch (error) {
      logger.error(`Failed to fetch paginated data for page ${page}:`, error);
      throw error;
    }
  }

  async loadInitialGraphData(): Promise<void> {
    try {
      // Try both endpoints
      const endpoints = [
        API_ENDPOINTS.GRAPH_PAGINATED
      ];

      let response = null;
      for (const endpoint of endpoints) {
        try {
          response = await fetch(`${endpoint}?page=1&pageSize=100`);
          if (response.ok) break;
        } catch (e) {
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
      
      // If there's only one page, enable randomization immediately
      if (data.totalPages <= 1) {
        const websocketService = WebSocketServiceClass.getInstance();
        logger.info('Single page graph data loaded. Enabling server-side randomization.');
        websocketService.enableRandomization(true);
      }
      
      logger.info('Initial graph data loaded successfully');
    } catch (error) {
      logger.error('Failed to fetch graph data:', error);
      throw new Error('Failed to fetch graph data: ' + error);
    }
  }

  private async loadRemainingPages(totalPages: number, pageSize: number): Promise<void> {
    try {
      throttledDebugLog(`Starting to load remaining pages. Total pages: ${totalPages}, Current nodes: ${this.nodes.size}`);
      
      // Load remaining pages in parallel with a reasonable chunk size
      const chunkSize = 5;
      for (let i = 2; i <= totalPages; i += chunkSize) {
        const pagePromises = [];
        for (let j = i; j < Math.min(i + chunkSize, totalPages + 1); j++) {
          pagePromises.push(this.loadPage(j, pageSize));
        }
        await Promise.all(pagePromises);
        // Update listeners after each chunk
        throttledDebugLog(`Loaded chunk ${i}-${Math.min(i + chunkSize - 1, totalPages)}. Current nodes: ${this.nodes.size}, edges: ${this.edges.size}`);
        this.notifyUpdateListeners();
      }
    } catch (error) {
      logger.error('Error loading remaining pages:', error);
      throw error;
    }
  }

  private async loadPage(page: number, pageSize: number): Promise<void> {
    try {
      throttledDebugLog(`Loading page ${page}. Current nodes before load: ${this.nodes.size}`);
      
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
      let newNodes = 0;
      transformedData.nodes.forEach((node: Node) => {
        if (!this.nodes.has(node.id)) {
          this.nodes.set(node.id, node);
          newNodes++;
        }
      });
      
      // Add new edges
      let newEdges = 0;
      transformedData.edges.forEach((edge: Edge) => {
        const edgeId = this.createEdgeId(edge.source, edge.target);
        if (!this.edges.has(edgeId)) {
          this.edges.set(edgeId, { ...edge, id: edgeId });
          newEdges++;
        }
      });

      throttledDebugLog(`Loaded page ${page}: ${newNodes} new nodes, ${newEdges} new edges. Total now: ${this.nodes.size} nodes, ${this.edges.size} edges`);
    } catch (error) {
      logger.error(`Error loading page ${page}:`, error);
      throw error;
    }
  }

  /**
   * Enable binary position updates via WebSocket
   */
  public enableBinaryUpdates(): void {
    // Enable binary updates flag - actual WebSocket connection is handled by WebSocketService
    this.setBinaryUpdatesEnabled(true);
    logger.info('Binary updates enabled');
  }

  /**
   * Enable or disable binary position updates
   */
  public setBinaryUpdatesEnabled(enabled: boolean): void {
    if (this.binaryUpdatesEnabled === enabled) return;
    
    this.binaryUpdatesEnabled = enabled;
    logger.info(`Binary updates ${enabled ? 'enabled' : 'disabled'}`);
    
    if (enabled) {
      // Check if WebSocket service is configured before sending update
      // Check if the send function is our default warning function
      const isDefaultService = this.wsService.send.toString().includes('WebSocket service not configured');
      if (!isDefaultService) {
        // Send initial empty update to start receiving binary updates
        this.updatePositions(new Float32Array());
      } else {
        logger.warn('Binary updates enabled but WebSocket service not yet configured. Will send update when service is available.');
        
        // Set up a retry mechanism to check for WebSocket service availability
        this.retryWebSocketConfiguration();
      }
    }
  }
  
  /**
   * Retry WebSocket configuration until it's available
   * This helps ensure we don't miss updates when the WebSocket service
   * is configured after binary updates are enabled
   */
  private retryWebSocketConfiguration(): void {
    // Only set up retry if not already running
    if (this._retryTimeout) {
      return;
    }
    
    const checkAndRetry = () => {
      // Check if WebSocket service is now configured
      const isDefaultService = this.wsService.send.toString().includes('WebSocket service not configured');
      if (!isDefaultService) {
        // WebSocket service is now configured, send initial update
        logger.info('WebSocket service now available, sending initial update');
        this.updatePositions(new Float32Array());
        this._retryTimeout = null;
      } else {
        // Still not configured, retry after delay
        this._retryTimeout = setTimeout(checkAndRetry, 1000) as any;
      }
    };
    
    // Start the retry process
    this._retryTimeout = setTimeout(checkAndRetry, 1000) as any;
  }
  
  private _retryTimeout: any = null;

  /**
   * Update node positions via binary protocol
   */
  private updatePositions(positions: Float32Array): void {
    if (!this.binaryUpdatesEnabled) {
      logger.warn('Attempted to update positions while binary updates are disabled');
      return;
    }
    
    try {
      // Check if WebSocket service is properly configured
      // Check if the send function is our default warning function
      const isDefaultService = this.wsService.send.toString().includes('WebSocket service not configured');
      if (isDefaultService) {
        logger.warn('Cannot send position update: WebSocket service not configured');
        // Set up retry mechanism if not already running
        this.retryWebSocketConfiguration();
        return;
      }
      
      this.wsService.send(positions.buffer);
    } catch (error) {
      logger.error('Failed to send position update:', error);
      // Don't disable binary updates on error - let the application decide
      // this.binaryUpdatesEnabled = false;
    }
  }

  /**
   * Initialize or update the graph data
   */
  updateGraphData(data: any): void {
    // Transform and validate incoming data
    const transformedData = transformGraphData(data);
    throttledDebugLog(`Updating graph data. Incoming: ${transformedData.nodes.length} nodes, ${transformedData.edges?.length || 0} edges. First 3 node IDs: ${transformedData.nodes.slice(0, 3).map(n => n.id).join(', ')}`);
    
    // Debug edge source/target IDs
    if (transformedData.edges && transformedData.edges.length > 0) {
      throttledDebugLog(`First 3 edge source/target IDs: ${transformedData.edges.slice(0, 3).map(e => `${e.source}->${e.target}`).join(', ')}`);
    }
    
    // Update nodes with proper position and velocity
    transformedData.nodes.forEach((node: Node) => {
      // Check if we already have this node
      const existingNode = this.nodes.get(node.id);
      
      
      if (existingNode) {
        // Update position and velocity
        existingNode.data.position.copy(node.data.position);
        if (node.data.velocity) {
          existingNode.data.velocity.copy(node.data.velocity);
        }
        
        // Only update metadata if the new node has valid metadata that's better than what we have
        if (node.data.metadata?.name && 
            node.data.metadata.name !== node.id && 
            node.data.metadata.name.length > 0) {
          existingNode.data.metadata = {
            ...existingNode.data.metadata,
            ...node.data.metadata
          };
        }
      } else {
        this.nodes.set(node.id, node);
      }
    });

    // Store edges in Map with generated IDs
    if (Array.isArray(transformedData.edges)) {
      transformedData.edges.forEach((edge: Edge) => {
        const edgeId = this.createEdgeId(edge.source, edge.target);
        
        // Check if source and target nodes exist
        if (!this.nodes.has(edge.source) || !this.nodes.has(edge.target)) {
          throttledDebugLog(`Skipping edge ${edge.source}->${edge.target} due to missing node(s)`);
          return;
        }
        
        const edgeWithId: EdgeWithId = {
          ...edge,
          id: edgeId
        };
        this.edges.set(edgeId, edgeWithId);
      });
    }

    // Update metadata, including pagination info if available
    this.metadata = {
      ...transformedData.metadata,
      pagination: data.totalPages ? {
        totalPages: data.totalPages,
        currentPage: data.currentPage,
        totalItems: data.totalItems,
        pageSize: data.pageSize
      } : undefined
    };

    // Notify listeners
    this.notifyUpdateListeners();
    throttledDebugLog(`Updated graph data: ${this.nodes.size} nodes, ${this.edges.size} edges`);

    // Enable binary updates after initial data is received
    if (!this.binaryUpdatesEnabled) {
      this.enableBinaryUpdates();
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

  public updateNodePositions(positions: Float32Array): void {
    if (!this.binaryUpdatesEnabled) return;
    
    const nodeCount = positions.length / FLOATS_PER_NODE;
    if (positions.length % FLOATS_PER_NODE !== 0) {
      logger.error('Invalid position array length:', positions.length);
      return;
    }

    // Buffer the updates
    for (let i = 0; i < nodeCount; i++) {
      const offset = i * FLOATS_PER_NODE;
      // Need to extract the node ID correctly - first 4 bytes in the array represent node ID
      // But in our current binary format, we need to get it from a numeric index
      const nodeId = i.toString(); // Convert numeric index to string ID
      
      if (!nodeId) continue;

      // Create proper THREE.Vector3 object instead of a plain object
      this.positionUpdateBuffer.set(nodeId, new Vector3(
        positions[offset],
        positions[offset + 1],
        positions[offset + 2]
      ));
    }

    // Schedule buffer flush if not already scheduled
    if (!this.updateBufferTimeout) {
      this.updateBufferTimeout = window.setTimeout(() => {
        this.flushPositionUpdates();
        this.updateBufferTimeout = null;
      }, GraphDataManager.BUFFER_FLUSH_INTERVAL);
    }
  }

  private flushPositionUpdates(): void {
    if (this.positionUpdateBuffer.size === 0) return;

    // Make sure we're working with proper THREE.Vector3 objects for data flow
    const updates = Array.from(this.positionUpdateBuffer.entries())
      .map(([id, position]) => ({
        id,
        data: { 
          position, // This is now a THREE.Vector3 object
          velocity: undefined 
        }
      }));

    // Convert node updates to Float32Array for binary protocol
    const nodesCount = updates.length;
    const positionsArray = new Float32Array(nodesCount * FLOATS_PER_NODE);
    
    updates.forEach((node, index) => {
      const baseIndex = index * FLOATS_PER_NODE;
      // Position (x, y, z)
      // Access x, y, z properties from the THREE.Vector3 object
      positionsArray[baseIndex] = node.data.position.x;
      positionsArray[baseIndex + 1] = node.data.position.y;
      positionsArray[baseIndex + 2] = node.data.position.z;
      // Velocity (set to 0 since undefined)
      positionsArray[baseIndex + 3] = positionsArray[baseIndex + 4] = positionsArray[baseIndex + 5] = 0;
    });
    this.notifyPositionUpdateListeners(positionsArray);
    this.positionUpdateBuffer.clear();
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
