import { defineStore } from 'pinia';
import type { VisualizationState, Node, Edge } from '../types/stores';

export const useVisualizationStore = defineStore('visualization', {
  state: (): VisualizationState => ({
    nodes: [],
    edges: [],
    metadata: {},
    isLoading: false,
    error: null,
    selectedNode: null,
    hoveredNode: null,
    cameraPosition: [0, 0, 100],
    cameraTarget: [0, 0, 0],
    renderSettings: {
      nodeSize: 5,
      nodeColor: '#1f77b4',
      edgeWidth: 1,
      edgeColor: '#999999',
      highlightColor: '#ff7f0e',
      opacity: 1,
      bloom: {
        enabled: false,
        strength: 1,
        radius: 1,
        threshold: 0.5
      },
      fisheye: {
        enabled: false,
        strength: 1,
        focusPoint: [0, 0, 0],
        radius: 100
      }
    },
    physicsSettings: {
      enabled: true,
      gravity: -1.2,
      springLength: 100,
      springStrength: 0.1,
      repulsion: 50,
      damping: 0.5,
      timeStep: 0.016
    }
  }),

  getters: {
    getNodeById: (state) => (id: string) => 
      state.nodes.find(node => node.id === id),

    getConnectedNodes: (state) => (nodeId: string) => {
      const connectedEdges = state.edges.filter(
        edge => edge.source === nodeId || edge.target === nodeId
      );
      const connectedNodeIds = new Set(
        connectedEdges.flatMap(edge => [edge.source, edge.target])
      );
      connectedNodeIds.delete(nodeId);
      return Array.from(connectedNodeIds);
    },

    getNodePosition: (state) => (nodeId: string) => {
      const node = state.nodes.find(n => n.id === nodeId);
      return node?.position || [0, 0, 0];
    },

    getVisibleNodes: (state) => {
      // TODO: Implement frustum culling logic
      return state.nodes;
    }
  },

  actions: {
    setGraphData(nodes: Node[], edges: Edge[], metadata: Record<string, any> = {}) {
      this.nodes = nodes;
      this.edges = edges;
      this.metadata = metadata;
    },

    updateNodePosition(nodeId: string, position: [number, number, number]) {
      const node = this.nodes.find(n => n.id === nodeId);
      if (node) {
        node.position = position;
      }
    },

    updateNodePositions(positions: Array<[string, [number, number, number]]>) {
      positions.forEach(([id, pos]) => {
        this.updateNodePosition(id, pos);
      });
    },

    selectNode(node: Node | null) {
      this.selectedNode = node;
    },

    setHoveredNode(node: Node | null) {
      this.hoveredNode = node;
    },

    setCameraPosition(position: [number, number, number]) {
      this.cameraPosition = position;
    },

    setCameraTarget(target: [number, number, number]) {
      this.cameraTarget = target;
    },

    updateRenderSettings(settings: Partial<VisualizationState['renderSettings']>) {
      this.renderSettings = { ...this.renderSettings, ...settings };
    },

    updatePhysicsSettings(settings: Partial<VisualizationState['physicsSettings']>) {
      this.physicsSettings = { ...this.physicsSettings, ...settings };
    },

    setError(error: string | null) {
      this.error = error;
    },

    setLoading(loading: boolean) {
      this.isLoading = loading;
    },

    reset() {
      this.nodes = [];
      this.edges = [];
      this.metadata = {};
      this.selectedNode = null;
      this.hoveredNode = null;
      this.error = null;
      this.isLoading = false;
      // Reset settings to defaults
      this.renderSettings = {
        nodeSize: 5,
        nodeColor: '#1f77b4',
        edgeWidth: 1,
        edgeColor: '#999999',
        highlightColor: '#ff7f0e',
        opacity: 1,
        bloom: {
          enabled: false,
          strength: 1,
          radius: 1,
          threshold: 0.5
        },
        fisheye: {
          enabled: false,
          strength: 1,
          focusPoint: [0, 0, 0],
          radius: 100
        }
      };
      this.physicsSettings = {
        enabled: true,
        gravity: -1.2,
        springLength: 100,
        springStrength: 0.1,
        repulsion: 50,
        damping: 0.5,
        timeStep: 0.016
      };
    }
  }
});
