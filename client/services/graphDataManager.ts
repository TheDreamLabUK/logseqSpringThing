import type WebsocketService from './websocketService'
import type { GraphUpdateMessage, BinaryMessage, Node as WSNode, Edge as WSEdge } from '../types/websocket'
import type { GraphNode, GraphEdge, GraphData } from '../types/core'

// Transform websocket node to graph node
const transformNode = (wsNode: WSNode): GraphNode => {
  console.debug('[GraphDataManager] Transforming node:', {
    id: wsNode.id,
    hasPosition: !!wsNode.position,
    position: wsNode.position,
    hasVelocity: !!wsNode.velocity,
    velocity: wsNode.velocity,
    timestamp: new Date().toISOString()
  });

  return {
    id: wsNode.id,
    label: wsNode.label || wsNode.id,
    position: wsNode.position,
    velocity: wsNode.velocity,
    size: wsNode.size,
    color: wsNode.color,
    type: wsNode.type,
    metadata: wsNode.metadata || {},
    userData: wsNode.userData || {},
    edges: [], // Will be populated after edges are transformed
    weight: wsNode.weight || 1,
    group: wsNode.group
  };
}

// Transform websocket edge to graph edge
const transformEdge = (sourceNode: GraphNode, targetNode: GraphNode, wsEdge: WSEdge): GraphEdge => {
  console.debug('[GraphDataManager] Transforming edge:', {
    source: wsEdge.source,
    target: wsEdge.target,
    sourceHasPosition: !!sourceNode.position,
    targetHasPosition: !!targetNode.position,
    timestamp: new Date().toISOString()
  });

  return {
    id: `${wsEdge.source}-${wsEdge.target}`,
    source: wsEdge.source,
    target: wsEdge.target,
    weight: wsEdge.weight || 1,
    width: wsEdge.width,
    color: wsEdge.color,
    type: wsEdge.type,
    metadata: wsEdge.metadata || {},
    userData: wsEdge.userData || {},
    sourceNode,
    targetNode,
    directed: wsEdge.directed || false
  };
}

export default class GraphDataManager {
  private websocketService: WebsocketService
  private graphData: GraphData | null = null
  // Map for quick node lookups by ID
  private nodeMap: Map<string, GraphNode> = new Map()

  constructor(websocketService: WebsocketService) {
    console.debug('[GraphDataManager] Initializing with websocket service');
    this.websocketService = websocketService
    
    // Set up event listeners with proper types
    this.websocketService.on('graphUpdate', this.handleGraphUpdate.bind(this))
    this.websocketService.on('gpuPositions', this.handleBinaryPositionUpdate.bind(this))

    // Debug listener for websocket connection state
    this.websocketService.on('open', () => {
      console.debug('[GraphDataManager] WebSocket connected, requesting initial data');
      this.requestInitialData()
    })
  }

  private handleGraphUpdate(message: GraphUpdateMessage) {
    console.debug('[GraphDataManager] Received graph update:', {
      hasGraphData: !!message.graphData,
      hasSnakeCaseData: !!message.graph_data,
      timestamp: new Date().toISOString()
    });

    if (!message.graphData && !message.graph_data) {
      console.warn('[GraphDataManager] Received graph update with no data');
      return
    }

    const graphData = message.graphData || message.graph_data;
    if (!graphData) return;

    console.debug('[GraphDataManager] Processing graph data:', {
      nodes: graphData.nodes?.length || 0,
      edges: graphData.edges?.length || 0,
      metadata: graphData.metadata ? Object.keys(graphData.metadata).length : 0,
      sampleNode: graphData.nodes?.[0] ? {
        id: graphData.nodes[0].id,
        hasPosition: !!graphData.nodes[0].position,
        position: graphData.nodes[0].position
      } : null
    });

    // Transform nodes first
    const nodes = (graphData.nodes || []).map(transformNode)
    
    // Create a map of nodes by ID for quick lookup
    this.nodeMap = new Map(nodes.map(node => [node.id, node]))

    console.debug('[GraphDataManager] Node map created:', {
      mapSize: this.nodeMap.size,
      sampleEntries: Array.from(this.nodeMap.entries()).slice(0, 3).map(([id, node]) => ({
        id,
        hasPosition: !!node.position,
        position: node.position
      }))
    });

    // Transform edges and link them to nodes
    const edges = (graphData.edges || []).map(edge => {
      const sourceNode = this.nodeMap.get(edge.source)
      const targetNode = this.nodeMap.get(edge.target)
      if (!sourceNode || !targetNode) {
        console.warn('[GraphDataManager] Edge references missing node:', {
          edge: `${edge.source}-${edge.target}`,
          hasSource: !!sourceNode,
          hasTarget: !!targetNode,
          timestamp: new Date().toISOString()
        })
        return null
      }
      const graphEdge = transformEdge(sourceNode, targetNode, edge)
      sourceNode.edges.push(graphEdge)
      targetNode.edges.push(graphEdge)
      return graphEdge
    }).filter((edge): edge is GraphEdge => edge !== null)

    // Store the transformed data
    this.graphData = {
      nodes,
      edges,
      metadata: graphData.metadata || {}
    }

    console.debug('[GraphDataManager] Graph data transformation complete:', {
      originalNodes: graphData.nodes?.length || 0,
      originalEdges: graphData.edges?.length || 0,
      transformedNodes: nodes.length,
      transformedEdges: edges.length,
      nodesWithPositions: nodes.filter(n => !!n.position).length,
      nodesWithVelocities: nodes.filter(n => !!n.velocity).length,
      timestamp: new Date().toISOString()
    });

    // Emit custom event for graph update
    window.dispatchEvent(new CustomEvent('graphData:update', {
      detail: this.graphData
    }))
  }

  private handleBinaryPositionUpdate(data: BinaryMessage) {
    if (!this.graphData?.nodes) {
      console.warn('[GraphDataManager] Received binary update but no graph data exists');
      return
    }

    console.debug('[GraphDataManager] Processing binary position update:', {
      bufferSize: data.data.byteLength,
      nodeCount: this.graphData.nodes.length,
      numMappedNodes: this.nodeMap.size,
      timestamp: new Date().toISOString()
    });

    // Process binary data
    const dataView = new Float32Array(data.data)
    const nodeCount = this.graphData.nodes.length
    
    // Update node positions from binary data
    let updatedCount = 0;
    let invalidCount = 0;

    for (let i = 0; i < nodeCount; i++) {
      const node = this.graphData.nodes[i]
      if (node) {
        const offset = i * 6;
        // Each node has 6 floats: x,y,z,vx,vy,vz
        const x = dataView[offset];
        const y = dataView[offset + 1];
        const z = dataView[offset + 2];
        const vx = dataView[offset + 3];
        const vy = dataView[offset + 4];
        const vz = dataView[offset + 5];

        // Validate values
        if (isNaN(x) || isNaN(y) || isNaN(z) || isNaN(vx) || isNaN(vy) || isNaN(vz)) {
          console.warn('[GraphDataManager] Invalid position/velocity values for node:', {
            id: node.id,
            position: [x, y, z],
            velocity: [vx, vy, vz]
          });
          invalidCount++;
          continue;
        }

        node.position = [x, y, z];
        node.velocity = [vx, vy, vz];
        updatedCount++;
      }
    }

    console.debug('[GraphDataManager] Binary update complete:', {
      totalNodes: nodeCount,
      updatedNodes: updatedCount,
      invalidNodes: invalidCount,
      timestamp: new Date().toISOString()
    });

    // Emit custom event for position update
    window.dispatchEvent(new CustomEvent('graphData:positions', {
      detail: {
        data: data.data,
        nodeCount,
        updatedCount,
        invalidCount
      }
    }))
  }

  public requestInitialData(): void {
    console.debug('[GraphDataManager] Requesting initial graph data');
    this.websocketService.send({ type: 'initial_data' })
  }

  public getGraphData(): GraphData | null {
    return this.graphData
  }

  public updateNodePosition(nodeId: string, position: [number, number, number]) {
    const node = this.nodeMap.get(nodeId)
    if (node) {
      console.debug('[GraphDataManager] Updating node position:', {
        id: nodeId,
        oldPosition: node.position,
        newPosition: position,
        timestamp: new Date().toISOString()
      });
      node.position = position
      this.websocketService.send({
        type: 'updateNodePosition',
        nodeId,
        position
      })
    }
  }

  public updateNodeVelocity(nodeId: string, velocity: [number, number, number]) {
    const node = this.nodeMap.get(nodeId)
    if (node) {
      console.debug('[GraphDataManager] Updating node velocity:', {
        id: nodeId,
        oldVelocity: node.velocity,
        newVelocity: velocity,
        timestamp: new Date().toISOString()
      });
      node.velocity = velocity
      this.websocketService.send({
        type: 'updateNodeVelocity',
        nodeId,
        velocity
      })
    }
  }

  public cleanup() {
    console.debug('[GraphDataManager] Cleaning up:', {
      hasGraphData: !!this.graphData,
      mapSize: this.nodeMap.size,
      timestamp: new Date().toISOString()
    });

    if (this.websocketService) {
      this.websocketService.off('graphUpdate', this.handleGraphUpdate.bind(this))
      this.websocketService.off('gpuPositions', this.handleBinaryPositionUpdate.bind(this))
    }
    this.graphData = null
    this.nodeMap.clear()
  }
}
