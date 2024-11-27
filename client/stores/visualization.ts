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
  selectedNode: Node | null
  metadata: Record<string, any>
  visualConfig: VisualizationConfig
  bloomConfig: BloomConfig
  fisheyeConfig: FisheyeConfig
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
    fisheyeConfig: { ...DEFAULT_FISHEYE_CONFIG }
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
  },

  actions: {
    setGraphData(nodes: Node[], edges: Edge[], metadata: Record<string, any> = {}) {
      console.log('Setting graph data:', {
        nodes: nodes.length,
        edges: edges.length,
        metadata: Object.keys(metadata).length
      })

      // Convert to graph data structure
      const graphNodes = nodes.map(node => ({
        ...node,
        edges: [],
        weight: node.weight || 1
      })) as GraphNode[]

      // Create node lookup for edge processing
      const nodeLookup = new Map<string, GraphNode>()
      graphNodes.forEach(node => nodeLookup.set(node.id, node))

      // Convert edges and link to nodes
      const graphEdges = edges.map(edge => {
        const sourceNode = nodeLookup.get(edge.source)
        const targetNode = nodeLookup.get(edge.target)
        if (!sourceNode || !targetNode) {
          console.warn('Edge references missing node:', edge)
          return null
        }
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

      console.log('Transformed graph data:', {
        graphNodes: graphNodes.length,
        graphEdges: graphEdges.length,
        sampleNode: graphNodes[0] ? {
          id: graphNodes[0].id,
          edges: graphNodes[0].edges.length,
          position: graphNodes[0].position
        } : null
      })

      this.nodes = nodes
      this.edges = edges
      this.metadata = metadata
      this.graphData = {
        nodes: graphNodes,
        edges: graphEdges,
        metadata
      }
    },

    updateNode(nodeId: string, updates: Partial<Node>) {
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
          }
        }
      }
    },

    updateNodePositions(updates: { id: string; position: [number, number, number]; velocity?: [number, number, number] }[]) {
      updates.forEach(update => {
        const node = this.nodes.find(n => n.id === update.id)
        if (node) {
          node.position = update.position
          if (update.velocity) {
            node.velocity = update.velocity
          }

          // Update graph data if it exists
          if (this.graphData) {
            const graphNode = this.graphData.nodes.find(n => n.id === update.id)
            if (graphNode) {
              graphNode.position = update.position
              if (update.velocity) {
                graphNode.velocity = update.velocity
              }
            }
          }
        }
      })
    },

    updateEdge(edgeId: string, updates: Partial<Edge>) {
      const index = this.edges.findIndex(e => e.id === edgeId)
      if (index !== -1) {
        this.edges[index] = { ...this.edges[index], ...updates }
        
        // Update graph data if it exists
        if (this.graphData) {
          const graphEdgeIndex = this.graphData.edges.findIndex(e => e.id === edgeId)
          if (graphEdgeIndex !== -1) {
            const graphEdge = this.graphData.edges[graphEdgeIndex]
            this.graphData.edges[graphEdgeIndex] = {
              ...graphEdge,
              ...updates,
              sourceNode: graphEdge.sourceNode,
              targetNode: graphEdge.targetNode
            }
          }
        }
      }
    },

    updateVisualizationSettings(settings: Partial<VisualizationConfig>) {
      console.log('Updating visualization settings:', settings)
      this.visualConfig = {
        ...this.visualConfig,
        ...settings
      }
    },

    updateBloomSettings(settings: Partial<BloomConfig>) {
      console.log('Updating bloom settings:', settings)
      this.bloomConfig = {
        ...this.bloomConfig,
        ...settings
      }
    },

    updateFisheyeSettings(settings: Partial<FisheyeConfig>) {
      console.log('Updating fisheye settings:', settings)
      // Convert focusPoint to individual coordinates if provided
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

    setSelectedNode(node: Node | null) {
      this.selectedNode = node
    },

    addNode(node: Node) {
      if (!this.nodes.find(n => n.id === node.id)) {
        this.nodes.push(node)
        if (this.graphData) {
          const graphNode: GraphNode = {
            ...node,
            edges: [],
            weight: node.weight || 1
          }
          this.graphData.nodes.push(graphNode)
        }
      }
    },

    removeNode(nodeId: string) {
      const index = this.nodes.findIndex(n => n.id === nodeId)
      if (index !== -1) {
        this.nodes.splice(index, 1)
        // Remove associated edges
        this.edges = this.edges.filter(edge => 
          edge.source !== nodeId && edge.target !== nodeId
        )
        
        // Update graph data if it exists
        if (this.graphData) {
          this.graphData.nodes = this.graphData.nodes.filter(n => n.id !== nodeId)
          this.graphData.edges = this.graphData.edges.filter(edge => 
            edge.sourceNode.id !== nodeId && edge.targetNode.id !== nodeId
          )
        }
      }
    },

    addEdge(edge: Edge) {
      if (!this.edges.find(e => e.id === edge.id)) {
        this.edges.push(edge)
        
        // Update graph data if it exists
        if (this.graphData) {
          const sourceNode = this.graphData.nodes.find(n => n.id === edge.source)
          const targetNode = this.graphData.nodes.find(n => n.id === edge.target)
          if (sourceNode && targetNode) {
            const graphEdge: GraphEdge = {
              ...edge,
              sourceNode,
              targetNode,
              directed: edge.directed || false
            }
            this.graphData.edges.push(graphEdge)
            sourceNode.edges.push(graphEdge)
            targetNode.edges.push(graphEdge)
          }
        }
      }
    },

    removeEdge(edgeId: string) {
      const index = this.edges.findIndex(e => e.id === edgeId)
      if (index !== -1) {
        this.edges.splice(index, 1)
        
        // Update graph data if it exists
        if (this.graphData) {
          this.graphData.edges = this.graphData.edges.filter(e => e.id !== edgeId)
        }
      }
    },

    clear() {
      this.nodes = []
      this.edges = []
      this.graphData = null
      this.selectedNode = null
      this.metadata = {}
      
      // Reset settings to defaults
      this.visualConfig = { ...DEFAULT_VISUALIZATION_CONFIG }
      this.bloomConfig = { ...DEFAULT_BLOOM_CONFIG }
      this.fisheyeConfig = { ...DEFAULT_FISHEYE_CONFIG }
    }
  }
})
