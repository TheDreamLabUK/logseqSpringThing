import { defineStore } from 'pinia'
import type { 
  Node, 
  Edge, 
  GraphNode,
  GraphEdge,
  GraphData,
  FisheyeSettings, 
  PhysicsSettings,
  MaterialSettings,
  BloomSettings,
  VisualizationSettings
} from '../types/core'

interface VisualizationState {
  nodes: Node[]
  edges: Edge[]
  graphData: GraphData | null
  selectedNode: Node | null
  metadata: Record<string, any>
  fisheyeSettings: FisheyeSettings
  physicsSettings: PhysicsSettings
  materialSettings: MaterialSettings
  bloomSettings: BloomSettings
}

export const useVisualizationStore = defineStore('visualization', {
  state: (): VisualizationState => ({
    nodes: [],
    edges: [],
    graphData: null,
    selectedNode: null,
    metadata: {},
    fisheyeSettings: {
      enabled: false,
      strength: 1.0,
      focusPoint: [0, 0, 0],
      radius: 100
    },
    physicsSettings: {
      enabled: true,
      gravity: -1.2,
      springLength: 100,
      springStrength: 0.1,
      repulsion: 1.0,
      damping: 0.5,
      timeStep: 0.016
    },
    materialSettings: {
      nodeSize: 1.0,
      nodeColor: '#ffffff',
      edgeWidth: 1.0,
      edgeColor: '#666666',
      highlightColor: '#ff0000',
      opacity: 1.0,
      metalness: 0.5,
      roughness: 0.5
    },
    bloomSettings: {
      enabled: true,
      strength: 1.5,
      radius: 0.4,
      threshold: 0.6
    }
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
    getFisheyeSettings: (state): FisheyeSettings => state.fisheyeSettings,
    getPhysicsSettings: (state): PhysicsSettings => state.physicsSettings,
    getMaterialSettings: (state): MaterialSettings => state.materialSettings,
    getBloomSettings: (state): BloomSettings => state.bloomSettings,
    
    getVisualizationSettings: (state): VisualizationSettings => ({
      material: state.materialSettings,
      physics: state.physicsSettings,
      bloom: state.bloomSettings,
      fisheye: state.fisheyeSettings
    })
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
        weight: 1
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
          directed: false
        }
        sourceNode.edges.push(edge)
        targetNode.edges.push(edge)
        return graphEdge
      }).filter((edge): edge is GraphEdge => edge !== null)

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
            this.graphData.nodes[graphNodeIndex] = {
              ...this.graphData.nodes[graphNodeIndex],
              ...updates
            }
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
        }
      })
    },

    updateEdge(edgeId: string, updates: Partial<Edge>) {
      const index = this.edges.findIndex(e => e.id === edgeId)
      if (index !== -1) {
        this.edges[index] = { ...this.edges[index], ...updates }
      }
    },

    updateFisheyeSettings(settings: Partial<FisheyeSettings>) {
      console.log('Updating fisheye settings:', settings)
      this.fisheyeSettings = {
        ...this.fisheyeSettings,
        ...settings
      }
    },

    updatePhysicsSettings(settings: Partial<PhysicsSettings>) {
      console.log('Updating physics settings:', settings)
      this.physicsSettings = {
        ...this.physicsSettings,
        ...settings
      }
    },

    updateMaterialSettings(settings: Partial<MaterialSettings>) {
      console.log('Updating material settings:', settings)
      this.materialSettings = {
        ...this.materialSettings,
        ...settings
      }
    },

    updateBloomSettings(settings: Partial<BloomSettings>) {
      console.log('Updating bloom settings:', settings)
      this.bloomSettings = {
        ...this.bloomSettings,
        ...settings
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
            weight: 1
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
              directed: false
            }
            this.graphData.edges.push(graphEdge)
            sourceNode.edges.push(edge)
            targetNode.edges.push(edge)
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
      this.fisheyeSettings = {
        enabled: false,
        strength: 1.0,
        focusPoint: [0, 0, 0],
        radius: 100
      }
      this.physicsSettings = {
        enabled: true,
        gravity: -1.2,
        springLength: 100,
        springStrength: 0.1,
        repulsion: 1.0,
        damping: 0.5,
        timeStep: 0.016
      }
      this.materialSettings = {
        nodeSize: 1.0,
        nodeColor: '#ffffff',
        edgeWidth: 1.0,
        edgeColor: '#666666',
        highlightColor: '#ff0000',
        opacity: 1.0,
        metalness: 0.5,
        roughness: 0.5
      }
      this.bloomSettings = {
        enabled: true,
        strength: 1.5,
        radius: 0.4,
        threshold: 0.6
      }
    }
  }
})
