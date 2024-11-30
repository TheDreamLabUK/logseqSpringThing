import type WebsocketService from './websocketService'
import type { GraphUpdateMessage, BinaryMessage } from '../types/websocket'
import type { GraphNode, GraphEdge, GraphData } from '../types/core'
import { useVisualizationStore } from '../stores/visualization'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'

/**
 * GraphDataManager handles rapid client-side interactions and WebSocket updates
 * 
 * Key responsibilities:
 * 1. Provides optimistic updates for user interactions before server confirmation
 * 2. Handles real-time position updates via WebSocket
 * 3. Maintains a lightweight cache for quick position lookups
 * 4. Transforms WebSocket messages to store-compatible formats
 */
export default class GraphDataManager {
  private websocketService: WebsocketService
  private nodePositionCache: Map<string, [number, number, number]> = new Map()
  private visualizationStore = useVisualizationStore()
  private binaryStore = useBinaryUpdateStore()

  constructor(websocketService: WebsocketService) {
    this.websocketService = websocketService
    
    // Set up WebSocket event listeners
    this.websocketService.on('graphUpdate', this.handleGraphUpdate.bind(this))
    this.websocketService.on('gpuPositions', this.handleBinaryPositionUpdate.bind(this))
    this.websocketService.on('open', () => {
      this.requestInitialData()
    })
  }

  private handleGraphUpdate(message: GraphUpdateMessage) {
    const graphData = message.graphData || message.graph_data
    if (!graphData) return

    // Update visualization store with new graph data
    this.visualizationStore.setGraphData(
      graphData.nodes || [],
      graphData.edges || [],
      graphData.metadata || {}
    )
    
    // Cache node positions for quick access
    graphData.nodes?.forEach(node => {
      if (node.position) {
        this.nodePositionCache.set(node.id, node.position)
      }
    })
  }

  private handleBinaryPositionUpdate(data: BinaryMessage) {
    // Update binary store with new position data
    this.binaryStore.updateFromBinary(data)

    // Update position cache from binary data
    const dataView = new Float32Array(data.data)
    const nodeCount = this.visualizationStore.nodes.length
    
    // Prepare batch updates for visualization store
    const updates: { id: string; position: [number, number, number]; velocity?: [number, number, number] }[] = []
    
    for (let i = 0; i < nodeCount; i++) {
      const node = this.visualizationStore.nodes[i]
      if (node) {
        const offset = i * 6
        const position: [number, number, number] = [
          dataView[offset],
          dataView[offset + 1],
          dataView[offset + 2]
        ]
        const velocity: [number, number, number] = [
          dataView[offset + 3],
          dataView[offset + 4],
          dataView[offset + 5]
        ]
        
        // Update cache
        this.nodePositionCache.set(node.id, position)
        
        // Add to batch update
        updates.push({
          id: node.id,
          position,
          velocity
        })
      }
    }
    
    // Batch update visualization store
    this.visualizationStore.updateNodePositions(updates)
  }

  /**
   * Optimistically update node position locally before server confirmation
   * This provides immediate feedback for user interactions
   */
  public updateNodePosition(nodeId: string, position: [number, number, number]) {
    // Update local cache immediately
    this.nodePositionCache.set(nodeId, position)
    
    // Update visualization store
    this.visualizationStore.updateNode(nodeId, { position })
    
    // Send update to server
    this.websocketService.send({
      type: 'updateNodePosition',
      nodeId,
      position
    })
  }

  /**
   * Get cached node position for rapid access
   * Falls back to store if position not in cache
   */
  public getNodePosition(nodeId: string): [number, number, number] | undefined {
    // Try cache first
    const cachedPosition = this.nodePositionCache.get(nodeId)
    if (cachedPosition) return cachedPosition
    
    // Fall back to store
    const node = this.visualizationStore.getNodeById(nodeId)
    return node?.position
  }

  public requestInitialData(): void {
    this.websocketService.send({ type: 'initial_data' })
  }

  public cleanup() {
    this.websocketService.off('graphUpdate', this.handleGraphUpdate.bind(this))
    this.websocketService.off('gpuPositions', this.handleBinaryPositionUpdate.bind(this))
    this.nodePositionCache.clear()
  }
}
