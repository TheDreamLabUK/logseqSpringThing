import { defineStore } from 'pinia'
import type { 
  Node, 
  Edge, 
  GraphNode,
  GraphEdge,
  GraphData,
  FisheyeSettings as CoreFisheyeSettings, 
  PhysicsSettings as CorePhysicsSettings,
  MaterialSettings as CoreMaterialSettings,
  BloomSettings as CoreBloomSettings
} from '../types/core'
import type {
  VisualizationConfig,
  BloomConfig,
  FisheyeConfig
} from '../types/components'
import {
  DEFAULT_VISUALIZATION_CONFIG,
  DEFAULT_BLOOM_CONFIG,
  DEFAULT_FISHEYE_CONFIG
} from '../types/components'

interface VisualizationState {
  nodes: Node[]
  edges: Edge[]
  graphData: GraphData | null
  selectedNode: null | Node
  metadata: Record<string, any>
  visualConfig: VisualizationConfig
  bloomConfig: BloomConfig
  fisheyeConfig: FisheyeConfig
  initialized: boolean
}

export const useVisualizationStore = defineStore('visualization', {
  state: (): VisualizationState => ({
    nodes: [],
    edges: [],
    graphData: null,
    selectedNode: null,
    metadata: {},
    visualConfig: { ...DEFAULT_VISUALIZATION_CONFIG },
    bloomConfig: { ...DEFAULT_BLOOM_CONFIG },
    fisheyeConfig: { ...DEFAULT_FISHEYE_CONFIG },
    initialized: false
  }),

  getters: {
    getNodeById: (state) => (id: string) => {
      return state.nodes.find(node => node.id === id)
    },

    getEdgesByNodeId: (state) => (nodeId: string) => {
      return state.edges.filter(edge => 
        edge.source === nodeId || edge.target === nodeId
      )
    },

    getGraphData: (state): GraphData | null => state.graphData,
    getVisualizationSettings: (state): VisualizationConfig => state.visualConfig,
    getBloomSettings: (state): BloomConfig => state.bloomConfig,
    getFisheyeSettings: (state): FisheyeConfig => state.fisheyeConfig,
    isInitialized: (state): boolean => state.initialized
  },

  actions: {
    setGraphData(nodes: Node[], edges: Edge[], metadata: Record<string, any> = {}) {
      console.debug('[VisualizationStore] Setting graph data:', {
        nodeCount: nodes.length,
        edgeCount: edges.length,
        metadataKeys: Object.keys(metadata),
        timestamp: new Date().toISOString(),
        initialized: this.initialized,
        sampleNodes: nodes.slice(0, 3).map(n => ({
          id: n.id,
          position: n.position,
          hasPosition: !!n.position,
          velocity: n.velocity,
          hasVelocity: !!n.velocity
        }))
      })

      // Validate node positions
      const nodesWithoutPosition = nodes.filter(n => !n.position)
      if (nodesWithoutPosition.length > 0) {
        console.warn('[VisualizationStore] Nodes missing position data:', {
          count: nodesWithoutPosition.length,
          sampleIds: nodesWithoutPosition.slice(0, 3).map(n => n.id),
          timestamp: new Date().toISOString()
        })
      }

      // Convert to graph data structure
      const graphNodes = nodes.map(node => ({
        ...node,
        edges: [],
        weight: node.weight || 1
      })) as GraphNode[]

      // Create node lookup for edge processing
      const nodeLookup = new Map<string, GraphNode>()
      graphNodes.forEach(node => nodeLookup.set(node.id, node))

      console.debug('[VisualizationStore] Node lookup created:', {
        lookupSize: nodeLookup.size,
        sampleEntries: Array.from(nodeLookup.entries()).slice(0, 3).map(([id, node]) => ({
          id,
          position: node.position,
          hasPosition: !!node.position,
          velocity: node.velocity,
          hasVelocity: !!node.velocity
        }))
      })

      // Track edge connection stats
      let missingSourceCount = 0
      let missingTargetCount = 0
      let validEdgeCount = 0

      // Convert edges and link to nodes
      const graphEdges = edges.map(edge => {
        const sourceNode = nodeLookup.get(edge.source)
        const targetNode = nodeLookup.get(edge.target)
        
        if (!sourceNode || !targetNode) {
          if (!sourceNode) missingSourceCount++
          if (!targetNode) missingTargetCount++
          
          console.warn('[VisualizationStore] Edge references missing node:', {
            edge: `${edge.source}-${edge.target}`,
            hasSource: !!sourceNode,
            hasTarget: !!targetNode,
            timestamp: new Date().toISOString()
          })
          return null
        }

        validEdgeCount++
        const graphEdge: GraphEdge = {
          ...edge,
          sourceNode,
          targetNode,
          directed: edge.directed || false
        }
        sourceNode.edges.push(graphEdge)
        targetNode.edges.push(graphEdge)
        return graphEdge
      }).filter((edge): edge is GraphEdge => edge !== null)

      console.debug('[VisualizationStore] Edge processing complete:', {
        totalEdges: edges.length,
        validEdges: validEdgeCount,
        missingSourceNodes: missingSourceCount,
        missingTargetNodes: missingTargetCount,
        timestamp: new Date().toISOString()
      })

      console.debug('[VisualizationStore] Graph data transformation complete:', {
        originalNodes: nodes.length,
        originalEdges: edges.length,
        transformedNodes: graphNodes.length,
        transformedEdges: graphEdges.length,
        nodesWithoutPosition: nodesWithoutPosition.length,
        sampleGraphNode: graphNodes[0] ? {
          id: graphNodes[0].id,
          edgeCount: graphNodes[0].edges.length,
          position: graphNodes[0].position,
          hasPosition: !!graphNodes[0].position,
          velocity: graphNodes[0].velocity,
          hasVelocity: !!graphNodes[0].velocity
        } : null,
        timestamp: new Date().toISOString()
      })

      // Store the data
      this.nodes = nodes
      this.edges = edges
      this.metadata = metadata
      this.graphData = {
        nodes: graphNodes,
        edges: graphEdges,
        metadata
      }
      this.initialized = true

      // Log final state
      console.debug('[VisualizationStore] Graph data state after update:', {
        storeNodes: this.nodes.length,
        storeEdges: this.edges.length,
        graphDataNodes: this.graphData.nodes.length,
        graphDataEdges: this.graphData.edges.length,
        initialized: this.initialized,
        timestamp: new Date().toISOString()
      })
    },

    updateNode(nodeId: string, updates: Partial<Node>) {
      console.debug('[VisualizationStore] Updating node:', {
        nodeId,
        updates,
        hasPosition: !!updates.position,
        hasVelocity: !!updates.velocity,
        timestamp: new Date().toISOString()
      })

      const index = this.nodes.findIndex(n => n.id === nodeId)
      if (index !== -1) {
        this.nodes[index] = { ...this.nodes[index], ...updates }
        
        // Update graph data if it exists
        if (this.graphData) {
          const graphNodeIndex = this.graphData.nodes.findIndex(n => n.id === nodeId)
          if (graphNodeIndex !== -1) {
            const graphNode = this.graphData.nodes[graphNodeIndex]
            this.graphData.nodes[graphNodeIndex] = {
              ...graphNode,
              ...updates,
              edges: graphNode.edges // Preserve edges array
            } as GraphNode

            console.debug('[VisualizationStore] Graph node updated:', {
              nodeId,
              position: updates.position,
              hasPosition: !!updates.position,
              velocity: updates.velocity,
              hasVelocity: !!updates.velocity,
              edgeCount: graphNode.edges.length,
              timestamp: new Date().toISOString()
            })
          }
        }
      } else {
        console.warn('[VisualizationStore] Node not found for update:', {
          nodeId,
          timestamp: new Date().toISOString()
        })
      }
    },

    updateNodePositions(updates: { id: string; position: [number, number, number]; velocity?: [number, number, number] }[]) {
      console.debug('[VisualizationStore] Batch updating node positions:', {
        updateCount: updates.length,
        timestamp: new Date().toISOString(),
        sampleUpdates: updates.slice(0, 3).map(u => ({
          id: u.id,
          position: u.position,
          hasVelocity: !!u.velocity
        }))
      })

      let updatedCount = 0
      let skippedCount = 0
      let missingPositionCount = 0

      updates.forEach(update => {
        if (!update.position) {
          missingPositionCount++
          return
        }

        const node = this.nodes.find(n => n.id === update.id)
        if (node) {
          // Update position and velocity directly (already scaled)
          node.position = update.position
          if (update.velocity) {
            node.velocity = update.velocity
          }

          // Update graph data if it exists
          if (this.graphData) {
            const graphNode = this.graphData.nodes.find(n => n.id === update.id)
            if (graphNode) {
              graphNode.position = node.position
              if (update.velocity) {
                graphNode.velocity = node.velocity
              }
              updatedCount++
            }
          }
        } else {
          skippedCount++
        }
      })

      console.debug('[VisualizationStore] Node position updates complete:', {
        totalUpdates: updates.length,
        successfulUpdates: updatedCount,
        skippedUpdates: skippedCount,
        missingPositions: missingPositionCount,
        timestamp: new Date().toISOString()
      })
    },

    updateVisualizationSettings(settings: Partial<VisualizationConfig>) {
      console.debug('[VisualizationStore] Updating visualization settings:', {
        oldSettings: this.visualConfig,
        newSettings: settings,
        timestamp: new Date().toISOString()
      })
      this.visualConfig = {
        ...this.visualConfig,
        ...settings
      }
    },

    updateBloomSettings(settings: Partial<BloomConfig>) {
      console.debug('[VisualizationStore] Updating bloom settings:', {
        oldSettings: this.bloomConfig,
        newSettings: settings,
        timestamp: new Date().toISOString()
      })
      this.bloomConfig = {
        ...this.bloomConfig,
        ...settings
      }
    },

    updateFisheyeSettings(settings: Partial<FisheyeConfig>) {
      console.debug('[VisualizationStore] Updating fisheye settings:', {
        oldSettings: this.fisheyeConfig,
        newSettings: settings,
        timestamp: new Date().toISOString()
      })
      
      if ('focusPoint' in settings) {
        const [focus_x, focus_y, focus_z] = settings.focusPoint as [number, number, number]
        this.fisheyeConfig = {
          ...this.fisheyeConfig,
          ...settings,
          focus_x,
          focus_y,
          focus_z
        }
      } else {
        this.fisheyeConfig = {
          ...this.fisheyeConfig,
          ...settings
        }
      }
    },

    clear() {
      console.debug('[VisualizationStore] Clearing visualization store:', {
        nodeCount: this.nodes.length,
        edgeCount: this.edges.length,
        hasGraphData: !!this.graphData,
        wasInitialized: this.initialized,
        timestamp: new Date().toISOString()
      })
      
      this.nodes = []
      this.edges = []
      this.graphData = null
      this.selectedNode = null
      this.metadata = {}
      this.initialized = false
      
      // Reset settings to defaults
      this.visualConfig = { ...DEFAULT_VISUALIZATION_CONFIG }
      this.bloomConfig = { ...DEFAULT_BLOOM_CONFIG }
      this.fisheyeConfig = { ...DEFAULT_FISHEYE_CONFIG }
    }
  }
})
