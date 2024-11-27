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
const transformEdge = (wsNode: WSNode, wsEdge: WSEdge): GraphEdge => {
  const sourceNode = transformNode(wsNode);
  const targetNode = transformNode(wsNode);
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

    // Transform nodes first
    const nodes = (message.graphData.nodes || []).map(transformNode)
    
    // Create a map of nodes by ID for quick lookup
    const nodeMap = new Map(nodes.map(node => [node.id, node]))

    // Transform edges and link them to nodes
    const edges = (message.graphData.edges || []).map(edge => {
      const sourceNode = nodeMap.get(edge.source)
      const targetNode = nodeMap.get(edge.target)
      if (!sourceNode || !targetNode) {
        console.warn(`Edge references missing node: ${edge.source} -> ${edge.target}`)
        return null
      }
      const graphEdge = transformEdge(sourceNode, edge)
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

    // Emit custom event for graph update
    window.dispatchEvent(new CustomEvent('graphData:update', {
      detail: this.graphData
    }))
  }

  private handleBinaryPositionUpdate(data: BinaryMessage) {
    if (!this.graphData?.nodes) return

    // Update node positions from binary data
    data.positions.forEach((pos, index) => {
      if (index < this.graphData!.nodes.length) {
        this.graphData!.nodes[index].position = [pos.x, pos.y, pos.z]
        this.graphData!.nodes[index].velocity = [pos.vx, pos.vy, pos.vz]
      }
    })

    // Emit custom event for position update
    window.dispatchEvent(new CustomEvent('graphData:positions', {
      detail: {
        positions: data.positions,
        isInitialLayout: data.isInitialLayout,
        timeStep: data.timeStep
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
    if (!this.graphData) return

    const node = this.graphData.nodes.find(n => n.id === nodeId)
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
    if (!this.graphData) return

    const node = this.graphData.nodes.find(n => n.id === nodeId)
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
  }
}
