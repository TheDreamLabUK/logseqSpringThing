// public/js/services/ragflowService.js

export class RAGflowService {
  constructor(websocketService) {
    this.websocketService = websocketService;
    this.setupWebSocketListeners();
    this.isConnected = false;
  }

  setupWebSocketListeners() {
    this.websocketService.on('open', () => {
      console.log('WebSocket connection established');
      this.isConnected = true;
      // Notify that the chat is ready
      const event = new CustomEvent('chatReady');
      window.dispatchEvent(event);
    });

    this.websocketService.on('close', () => {
      console.log('WebSocket connection closed');
      this.isConnected = false;
    });

    this.websocketService.on('message', (data) => {
      console.log('Received message:', data);
      switch (data.type) {
        case 'ragflowResponse':
          this.handleRAGFlowResponse(data);
          break;
        case 'chatHistoryResponse':
          this.handleChatHistoryResponse(data);
          break;
        case 'error':
          this.handleError(data);
          break;
        default:
          console.warn('Unhandled message type:', data.type);
      }
    });
  }

  checkConnection() {
    if (!this.isConnected) {
      console.error('WebSocket is not connected');
      return false;
    }
    return true;
  }

  sendQuery(query, quote = false, docIds = null, stream = false) {
    if (!this.checkConnection()) return;

    console.log('Sending query:', query);
    this.websocketService.send({
      type: 'ragflowQuery',
      message: query,
      quote: quote,
      docIds: docIds,
      stream: stream
    });
  }

  handleRAGFlowResponse(data) {
    console.log('Received RAGFlow response:', data);
    if (data.data) {
      let answer, reference;
      if (data.data.answer) {
        // Handle response with answer field
        answer = data.data.answer;
        reference = data.data.reference || [];
      } else if (Array.isArray(data.data.message)) {
        // Handle message array (even if empty)
        answer = data.data.message.length > 0 ? data.data.message[0].content : "No response received from the server.";
        reference = [];
      } else {
        console.error('Unexpected RAGFlow response format:', data);
        this.handleError({ message: 'Unexpected response format from server' });
        return;
      }

      const event = new CustomEvent('ragflowAnswer', { 
        detail: {
          messages: [{role: 'assistant', content: answer}],
          reference: reference
        }
      });
      window.dispatchEvent(event);
    } else {
      console.error('Invalid RAGFlow response:', data);
      this.handleError({ message: 'Invalid response from server' });
    }
  }

  getChatHistory() {
    if (!this.checkConnection()) return;

    console.log('Requesting chat history');
    this.websocketService.send({
      type: 'chatHistory'
    });
  }

  handleChatHistoryResponse(data) {
    console.log('Received chat history:', data);
    if (data.data && Array.isArray(data.data.message)) {
      const event = new CustomEvent('chatHistoryReceived', { 
        detail: {
          messages: data.data.message
        }
      });
      window.dispatchEvent(event);
    } else {
      console.error('Unexpected chat history format:', data);
      this.handleError({ message: 'Unexpected chat history format from server' });
    }
  }

  handleError(data) {
    console.error('RAGFlow error:', data.message || 'Unknown error');
    const event = new CustomEvent('ragflowError', { detail: data.message || 'Unknown error' });
    window.dispatchEvent(event);
  }
}
