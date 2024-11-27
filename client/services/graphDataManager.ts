import type WebsocketService from './websocketService'
import type { GraphUpdateMessage, BinaryMessage, Node as WSNode, Edge as WSEdge } from '../types/websocket'
import type { GraphNode, GraphEdge, GraphData } from '../types/core'

// Transform websocket node to graph node
const transformNode = (wsNode: WSNode): GraphNode => ({
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
})

// Transform websocket edge to graph edge
const transformEdge = (sourceNode: GraphNode, targetNode: GraphNode, wsEdge: WSEdge): GraphEdge => {
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
    this.websocketService = websocketService
    
    // Set up event listeners with proper types
    this.websocketService.on('graphUpdate', this.handleGraphUpdate.bind(this))
    this.websocketService.on('gpuPositions', this.handleBinaryPositionUpdate.bind(this))

    // Debug listener for websocket connection state
    this.websocketService.on('open', () => {
      console.log('GraphDataManager detected websocket connection')
      console.log('Requesting initial data')
      this.requestInitialData()
    })
  }

  private handleGraphUpdate(message: GraphUpdateMessage) {
    if (!message.graphData) {
      console.warn('Received graph update with no data')
      return
    }

    console.log('Received graph update:', {
      nodes: message.graphData.nodes?.length || 0,
      edges: message.graphData.edges?.length || 0,
      metadata: message.graphData.metadata ? Object.keys(message.graphData.metadata).length : 0
    })

    // Transform nodes first
    const nodes = (message.graphData.nodes || []).map(transformNode)
    
    // Create a map of nodes by ID for quick lookup
    this.nodeMap = new Map(nodes.map(node => [node.id, node]))

    // Transform edges and link them to nodes
    const edges = (message.graphData.edges || []).map(edge => {
      const sourceNode = this.nodeMap.get(edge.source)
      const targetNode = this.nodeMap.get(edge.target)
      if (!sourceNode || !targetNode) {
        console.warn(`Edge references missing node: ${edge.source} -> ${edge.target}`)
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
      metadata: message.graphData.metadata || {}
    }

    console.log('Graph data transformed:', {
      nodes: this.graphData.nodes.length,
      edges: this.graphData.edges.length,
      metadata: Object.keys(this.graphData.metadata).length
    })

    // Emit custom event for graph update
    window.dispatchEvent(new CustomEvent('graphData:update', {
      detail: this.graphData
    }))
  }

  private handleBinaryPositionUpdate(data: BinaryMessage) {
    if (!this.graphData?.nodes) {
      console.warn('Received binary update but no graph data exists')
      return
    }

    // Log binary update stats
    console.debug('Processing binary position update:', {
      numPositions: data.positions.length,
      isInitialLayout: data.isInitialLayout,
      numNodes: this.graphData.nodes.length,
      numMappedNodes: this.nodeMap.size
    })

    // Update node positions from binary data using node IDs
    data.positions.forEach((pos) => {
      const node = this.nodeMap.get(pos.id)
      if (node) {
        node.position = [pos.x, pos.y, pos.z]
        node.velocity = [pos.vx, pos.vy, pos.vz]
      } else {
        console.warn(`No node found for ID: ${pos.id}`)
      }
    })

    // Log sample of updated positions
    if (data.positions.length > 0) {
      const sampleNode = this.nodeMap.get(data.positions[0].id)
      console.debug('Sample node update:', {
        id: data.positions[0].id,
        position: sampleNode?.position,
        velocity: sampleNode?.velocity
      })
    }

    // Emit custom event for position update
    window.dispatchEvent(new CustomEvent('graphData:positions', {
      detail: {
        positions: data.positions,
        isInitialLayout: data.isInitialLayout
      }
    }))
  }

  public requestInitialData(): void {
    console.log('Requesting initial graph data')
    this.websocketService.send({ type: 'getInitialData' })
  }

  public getGraphData(): GraphData | null {
    return this.graphData
  }

  public updateNodePosition(nodeId: string, position: [number, number, number]) {
    const node = this.nodeMap.get(nodeId)
    if (node) {
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
      node.velocity = velocity
      this.websocketService.send({
        type: 'updateNodeVelocity',
        nodeId,
        velocity
      })
    }
  }

  public cleanup() {
    if (this.websocketService) {
      this.websocketService.off('graphUpdate', this.handleGraphUpdate.bind(this))
      this.websocketService.off('gpuPositions', this.handleBinaryPositionUpdate.bind(this))
    }
    this.graphData = null
    this.nodeMap.clear()
  }
}
