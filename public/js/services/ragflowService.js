export class RAGflowService {
  constructor(socket) {
    this.socket = socket;
  }

  sendQuery(query) {
    this.socket.send(JSON.stringify({
      type: 'ragflowQuery',
      question: query
    }));
  }

  handleResponse(responseData, callback) {
    const response = JSON.parse(responseData);
    if (response.type === 'ragflowResponse') {
      callback(response.response);
    }
  }

  handleError(error, errorCallback) {
    errorCallback('Error: Unable to connect to the server. Please try again later.');
  }

  handleClose(closeCallback) {
    closeCallback('Connection lost. Attempting to reconnect...');
  }

  setupWebSocket(handleResponse, handleError, handleClose) {
    this.socket.onmessage = (event) => {
      this.handleResponse(event.data, handleResponse);
    };

    this.socket.onerror = (error) => {
      this.handleError(error, handleError);
    };

    this.socket.onclose = () => {
      this.handleClose(handleClose);
    };
  }
}