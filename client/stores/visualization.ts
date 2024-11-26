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
    cameraTarget: [0, 0, 0]
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

    selectNode(nodeId: string | null) {
      this.selectedNode = nodeId;
    },

    setHoveredNode(nodeId: string | null) {
      this.hoveredNode = nodeId;
    },

    setCameraPosition(position: [number, number, number]) {
      this.cameraPosition = position;
    },

    setCameraTarget(target: [number, number, number]) {
      this.cameraTarget = target;
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
    }
  }
});
