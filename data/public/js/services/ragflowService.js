// public/js/services/ragflowService.js

/**
 * RAGflowService manages interactions with the RAGFlow API for AI-powered question answering.
 */
export class RAGflowService {
  /**
   * Creates a new RAGflowService instance.
   * @param {WebsocketService} websocketService - The WebSocket service instance.
   */
  constructor(websocketService) {
      this.websocketService = websocketService;
      this.setupWebSocketListeners();
  }

  /**
   * Sets up WebSocket listeners specific to RAGFlow interactions.
   */
  setupWebSocketListeners() {
      // Listen for RAGFlow responses from the server
      this.websocketService.on('message', (data) => {
          if (data.type === 'ragflowResponse') {
              this.handleRAGFlowResponse(data);
          }
      });
  }

  /**
   * Sends a query to RAGFlow via WebSocket.
   * @param {string} query - The user's question.
   */
  sendQuery(query) {
      this.websocketService.send({
          type: 'ragflowQuery',
          question: query
      });
  }

  /**
   * Handles responses from RAGFlow.
   * @param {object} data - The response data from the server.
   */
  handleRAGFlowResponse(data) {
      const { answer } = data;
      // Dispatch a custom event to pass the answer to ChatManager
      const event = new CustomEvent('ragflowAnswer', { detail: answer });
      window.dispatchEvent(event);
  }
}
