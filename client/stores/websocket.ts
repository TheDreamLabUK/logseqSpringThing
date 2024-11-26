import { defineStore } from 'pinia';
import type { WebSocketState } from '../types/stores';
import type { 
  BaseMessage, 
  GraphUpdateMessage, 
  ErrorMessage,
  MessageType,
  Node as WebSocketNode,
  Edge as WebSocketEdge
} from '../types/websocket';
import type {
  VisualizationConfig,
  BloomConfig,
  FisheyeConfig
} from '../types/components';
import type { Node, Edge } from '../types/core';
import { useVisualizationStore } from './visualization';
import { useSettingsStore } from './settings';

export const useWebSocketStore = defineStore('websocket', {
  state: (): WebSocketState => ({
    isConnected: false,
    reconnectAttempts: 0,
    messageQueue: [],
    lastError: null,
    graphData: null
  }),

  getters: {
    connected: (state) => state.isConnected,
    hasError: (state) => state.lastError !== null,
    queueLength: (state) => state.messageQueue.length
  },

  actions: {
    setConnected(connected: boolean) {
      this.isConnected = connected;
      if (connected) {
        this.reconnectAttempts = 0;
        this.processMessageQueue();
      }
    },

    incrementReconnectAttempts() {
      this.reconnectAttempts++;
    },

    setError(error: string | null) {
      this.lastError = error;
    },

    queueMessage(message: BaseMessage) {
      this.messageQueue.push(message);
    },

    processMessageQueue() {
      if (!this.isConnected) return;
      
      while (this.messageQueue.length > 0) {
        const message = this.messageQueue.shift();
        if (message) {
          this.sendMessage(message);
        }
      }
    },

    transformNode(wsNode: WebSocketNode): Node {
      return {
        id: wsNode.id,
        label: wsNode.label || wsNode.id, // Use id as fallback if label is missing
        position: wsNode.position,
        velocity: wsNode.velocity,
        size: wsNode.size,
        color: wsNode.color,
        type: wsNode.type,
        metadata: wsNode.metadata || {},
        userData: wsNode.userData
      };
    },

    transformEdge(wsEdge: WebSocketEdge): Edge {
      return {
        id: wsEdge.id || `${wsEdge.source}-${wsEdge.target}`, // Generate id if missing
        source: wsEdge.source,
        target: wsEdge.target,
        weight: wsEdge.weight,
        width: wsEdge.width,
        color: wsEdge.color,
        type: wsEdge.type,
        metadata: wsEdge.metadata || {},
        userData: wsEdge.userData
      };
    },

    async handleMessage(message: BaseMessage) {
      const visualizationStore = useVisualizationStore();
      const settingsStore = useSettingsStore();

      try {
        switch (message.type as MessageType) {
          case 'graphUpdate':
          case 'graphData': {
            const graphMessage = message as GraphUpdateMessage;
            const transformedNodes = graphMessage.graphData.nodes.map(node => this.transformNode(node));
            const transformedEdges = graphMessage.graphData.edges.map(edge => this.transformEdge(edge));
            
            visualizationStore.setGraphData(
              transformedNodes,
              transformedEdges,
              graphMessage.graphData.metadata || {}
            );
            this.graphData = {
              nodes: transformedNodes,
              edges: transformedEdges,
              metadata: graphMessage.graphData.metadata || {}
            };
            break;
          }

          case 'error': {
            const errorMessage = message as ErrorMessage;
            this.setError(errorMessage.message);
            console.error('WebSocket error:', errorMessage.message);
            break;
          }

          case 'settings_updated': {
            const settings = message.settings as {
              visualization?: Partial<VisualizationConfig>;
              bloom?: Partial<BloomConfig>;
              fisheye?: Partial<FisheyeConfig>;
            };
            
            if (settings) {
              settingsStore.applyServerSettings(settings);
            }
            break;
          }

          default:
            console.warn('Unhandled message type:', message.type);
        }
      } catch (error) {
        console.error('Error handling message:', error);
        this.setError(error instanceof Error ? error.message : 'Unknown error');
      }
    },

    sendMessage(message: BaseMessage) {
      if (!this.isConnected) {
        this.queueMessage(message);
        return;
      }

      try {
        window.dispatchEvent(new CustomEvent('websocket:send', {
          detail: message
        }));
      } catch (error) {
        console.error('Error sending message:', error);
        this.setError(error instanceof Error ? error.message : 'Failed to send message');
        this.queueMessage(message);
      }
    },

    updateSettings(settings: {
      visualization?: Partial<VisualizationConfig>;
      bloom?: Partial<BloomConfig>;
      fisheye?: Partial<FisheyeConfig>;
    }) {
      const updateMessage: BaseMessage = {
        type: 'updateSettings',
        settings
      };
      this.sendMessage(updateMessage);
    },

    requestInitialData() {
      this.sendMessage({
        type: 'getInitialData'
      });
    },

    reset() {
      this.isConnected = false;
      this.reconnectAttempts = 0;
      this.messageQueue = [];
      this.lastError = null;
      this.graphData = null;
    }
  }
});
