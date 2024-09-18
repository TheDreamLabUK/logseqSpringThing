import { RAGflowService } from '../../public/js/services/ragflowService';

describe('RAGflowService', () => {
  let ragflowService;
  let mockSocket;

  beforeEach(() => {
    mockSocket = {
      send: jest.fn(),
      onmessage: null,
      onerror: null,
      onclose: null
    };
    ragflowService = new RAGflowService(mockSocket);
  });

  test('RAGflowService initializes correctly', () => {
    expect(ragflowService.socket).toBe(mockSocket);
  });

  test('sendQuery sends a query through WebSocket', () => {
    const query = 'What is RAG?';
    ragflowService.sendQuery(query);

    expect(mockSocket.send).toHaveBeenCalledWith(JSON.stringify({
      type: 'ragflowQuery',
      question: query
    }));
  });

  test('handleResponse processes RAGflow response correctly', () => {
    const mockResponse = {
      type: 'ragflowResponse',
      response: 'RAG stands for Retrieval-Augmented Generation.'
    };
    const mockCallback = jest.fn();

    ragflowService.handleResponse(JSON.stringify(mockResponse), mockCallback);

    expect(mockCallback).toHaveBeenCalledWith('RAG stands for Retrieval-Augmented Generation.');
  });

  test('handleResponse ignores non-RAGflow responses', () => {
    const mockResponse = {
      type: 'otherResponse',
      data: 'Some other data'
    };
    const mockCallback = jest.fn();

    ragflowService.handleResponse(JSON.stringify(mockResponse), mockCallback);

    expect(mockCallback).not.toHaveBeenCalled();
  });

  test('handleError calls error callback', () => {
    const mockError = new Error('WebSocket error');
    const mockErrorCallback = jest.fn();

    ragflowService.handleError(mockError, mockErrorCallback);

    expect(mockErrorCallback).toHaveBeenCalledWith('Error: Unable to connect to the server. Please try again later.');
  });

  test('handleClose calls close callback', () => {
    const mockCloseCallback = jest.fn();

    ragflowService.handleClose(mockCloseCallback);

    expect(mockCloseCallback).toHaveBeenCalledWith('Connection lost. Attempting to reconnect...');
  });

  test('setupWebSocket sets up WebSocket event handlers', () => {
    const mockHandleResponse = jest.fn();
    const mockHandleError = jest.fn();
    const mockHandleClose = jest.fn();

    ragflowService.setupWebSocket(mockHandleResponse, mockHandleError, mockHandleClose);

    expect(typeof mockSocket.onmessage).toBe('function');
    expect(typeof mockSocket.onerror).toBe('function');
    expect(typeof mockSocket.onclose).toBe('function');

    // Test onmessage handler
    const mockMessage = { data: JSON.stringify({ type: 'ragflowResponse', response: 'Test response' }) };
    mockSocket.onmessage(mockMessage);
    expect(mockHandleResponse).toHaveBeenCalledWith('Test response');

    // Test onerror handler
    const mockError = new Error('WebSocket error');
    mockSocket.onerror(mockError);
    expect(mockHandleError).toHaveBeenCalled();

    // Test onclose handler
    mockSocket.onclose();
    expect(mockHandleClose).toHaveBeenCalled();
  });
});
