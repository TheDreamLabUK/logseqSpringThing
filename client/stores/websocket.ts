import { defineStore } from 'pinia';
import type { WebSocketState } from '@/types/stores';
import type { 
  BaseMessage, 
  GraphUpdateMessage, 
  ErrorMessage,
  MessageType
} from '@/types/websocket';
import type {
  VisualizationConfig,
  BloomConfig,
  FisheyeConfig
} from '@/types/components';
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

    async handleMessage(message: BaseMessage) {
      const visualizationStore = useVisualizationStore();
      const settingsStore = useSettingsStore();

      try {
        switch (message.type as MessageType) {
          case 'graphUpdate':
          case 'graphData': {
            const graphMessage = message as GraphUpdateMessage;
            visualizationStore.setGraphData(
              graphMessage.graphData.nodes,
              graphMessage.graphData.edges,
              graphMessage.graphData.metadata
            );
            this.graphData = graphMessage.graphData;
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
