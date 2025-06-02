/**
 * Graph Data Web Worker
 * Handles graph data processing, binary decompression, and position updates off the main thread
 */

import { expose } from 'comlink';
import { BinaryNodeData, parseBinaryNodeData, createBinaryNodeData, Vec3 } from '../../../types/binaryProtocol';

export interface Node {
  id: string;
  label: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  metadata?: Record<string, any>;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  label?: string;
  weight?: number;
  metadata?: Record<string, any>;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

/**
 * Decompress zlib compressed data in worker thread
 */
async function decompressZlib(compressedData: ArrayBuffer): Promise<ArrayBuffer> {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      const cs = new DecompressionStream('deflate-raw');
      const writer = cs.writable.getWriter();
      writer.write(new Uint8Array(compressedData.slice(2))); // Skip zlib header
      writer.close();
      
      const output = [];
      const reader = cs.readable.getReader();
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        output.push(value);
      }
      
      const totalLength = output.reduce((acc, arr) => acc + arr.length, 0);
      const result = new Uint8Array(totalLength);
      let offset = 0;
      
      for (const arr of output) {
        result.set(arr, offset);
        offset += arr.length;
      }
      
      return result.buffer;
    } catch (error) {
      console.error('Worker decompression failed:', error);
      throw error;
    }
  }
  throw new Error('DecompressionStream not available');
}

/**
 * Check if data is zlib compressed
 */
function isZlibCompressed(data: ArrayBuffer): boolean {
  if (data.byteLength < 2) return false;
  const view = new Uint8Array(data);
  return view[0] === 0x78 && [0x01, 0x5E, 0x9C, 0xDA].includes(view[1]);
}

/**
 * Graph Worker API exposed to main thread
 */
class GraphWorker {
  private graphData: GraphData = { nodes: [], edges: [] };
  private nodeIdMap: Map<string, number> = new Map();
  private reverseNodeIdMap: Map<number, string> = new Map();
  private positionBuffer: SharedArrayBuffer | null = null;
  private positionView: Float32Array | null = null;

  /**
   * Initialize the worker with graph data
   */
  async setGraphData(data: GraphData): Promise<void> {
    this.graphData = {
      nodes: data.nodes.map(node => this.ensureNodeHasValidPosition(node)),
      edges: data.edges
    };
    
    // Create ID mappings
    this.nodeIdMap.clear();
    this.reverseNodeIdMap.clear();
    
    this.graphData.nodes.forEach((node, index) => {
      const numericId = parseInt(node.id, 10);
      if (!isNaN(numericId) && numericId >= 0 && numericId <= 0xFFFFFFFF) {
        this.nodeIdMap.set(node.id, numericId);
        this.reverseNodeIdMap.set(numericId, node.id);
      } else {
        const mappedId = index + 1;
        this.nodeIdMap.set(node.id, mappedId);
        this.reverseNodeIdMap.set(mappedId, node.id);
      }
    });

    console.log(`GraphWorker: Initialized with ${this.graphData.nodes.length} nodes`);
  }

  /**
   * Set up shared array buffer for position data
   */
  async setupSharedPositions(buffer: SharedArrayBuffer): Promise<void> {
    this.positionBuffer = buffer;
    this.positionView = new Float32Array(buffer);
    console.log(`GraphWorker: SharedArrayBuffer set up with ${buffer.byteLength} bytes`);
  }

  /**
   * Process binary position data with decompression
   */
  async processBinaryData(data: ArrayBuffer): Promise<Float32Array> {
    try {
      // Decompress if needed
      if (isZlibCompressed(data)) {
        data = await decompressZlib(data);
      }

      // Parse binary data
      const nodeUpdates = parseBinaryNodeData(data);
      
      if (nodeUpdates.length === 0) {
        console.warn('No valid node updates parsed from binary data');
        return new Float32Array(0);
      }

      // Create position array (4 values per node: id, x, y, z)
      const positionArray = new Float32Array(nodeUpdates.length * 4);
      let updatedCount = 0;

      nodeUpdates.forEach((nodeUpdate, index) => {
        const { nodeId, position, velocity } = nodeUpdate;
        const stringNodeId = this.reverseNodeIdMap.get(nodeId);
        
        if (stringNodeId) {
          const nodeIndex = this.graphData.nodes.findIndex(node => node.id === stringNodeId);
          if (nodeIndex >= 0) {
            const oldNode = this.graphData.nodes[nodeIndex];
            this.graphData.nodes[nodeIndex] = {
              ...oldNode,
              position: { ...position },
              metadata: {
                ...oldNode.metadata,
                velocity: { ...velocity }
              }
            };
            updatedCount++;
          }
        }
        
        // Update position array
        const arrayOffset = index * 4;
        positionArray[arrayOffset] = nodeId;
        positionArray[arrayOffset + 1] = position.x;
        positionArray[arrayOffset + 2] = position.y;
        positionArray[arrayOffset + 3] = position.z;
      });

      // Update shared buffer if available
      if (this.positionView && positionArray.length <= this.positionView.length) {
        this.positionView.set(positionArray);
      }

      console.log(`GraphWorker: Updated ${updatedCount} nodes from binary data`);
      return positionArray;
    } catch (error) {
      console.error('GraphWorker: Error processing binary data:', error);
      throw error;
    }
  }

  /**
   * Get current graph data
   */
  async getGraphData(): Promise<GraphData> {
    return this.graphData;
  }

  /**
   * Add or update a node
   */
  async updateNode(node: Node): Promise<void> {
    const existingIndex = this.graphData.nodes.findIndex(n => n.id === node.id);
    
    if (existingIndex >= 0) {
      this.graphData.nodes[existingIndex] = { ...this.graphData.nodes[existingIndex], ...node };
    } else {
      this.graphData.nodes.push(this.ensureNodeHasValidPosition(node));
      
      // Update ID mappings
      const numericId = parseInt(node.id, 10);
      if (!isNaN(numericId)) {
        this.nodeIdMap.set(node.id, numericId);
        this.reverseNodeIdMap.set(numericId, node.id);
      } else {
        const mappedId = this.graphData.nodes.length;
        this.nodeIdMap.set(node.id, mappedId);
        this.reverseNodeIdMap.set(mappedId, node.id);
      }
    }
  }

  /**
   * Remove a node
   */
  async removeNode(nodeId: string): Promise<void> {
    const numericId = this.nodeIdMap.get(nodeId);
    
    this.graphData.nodes = this.graphData.nodes.filter(node => node.id !== nodeId);
    this.graphData.edges = this.graphData.edges.filter(
      edge => edge.source !== nodeId && edge.target !== nodeId
    );
    
    if (numericId !== undefined) {
      this.nodeIdMap.delete(nodeId);
      this.reverseNodeIdMap.delete(numericId);
    }
  }

  /**
   * Create binary node data
   */
  async createBinaryData(nodes: BinaryNodeData[]): Promise<ArrayBuffer> {
    return createBinaryNodeData(nodes);
  }

  private ensureNodeHasValidPosition(node: Node): Node {
    if (!node.position) {
      return { ...node, position: { x: 0, y: 0, z: 0 } };
    }
    
    return {
      ...node,
      position: {
        x: typeof node.position.x === 'number' ? node.position.x : 0,
        y: typeof node.position.y === 'number' ? node.position.y : 0,
        z: typeof node.position.z === 'number' ? node.position.z : 0
      }
    };
  }
}

// Expose the worker API using Comlink
const worker = new GraphWorker();
expose(worker);

export type GraphWorkerType = GraphWorker;