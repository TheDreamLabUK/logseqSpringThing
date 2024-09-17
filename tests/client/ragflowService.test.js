import { RagflowService } from '../../public/js/services/ragflowService';

describe('RAGFlowService', () => {
  let ragflowService;

  beforeEach(() => {
    ragflowService = new RagflowService();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    global.fetch.mockClear();
    delete global.fetch;
  });

  test('should initialize properly', () => {
    expect(ragflowService).toBeInstanceOf(RagflowService);
  });

  test('should create a new conversation', async () => {
    global.fetch.mockResolvedValueOnce({
      json: () => Promise.resolve({ conversationId: '12345' }),
    });

    const conversationId = await ragflowService.createConversation('user1');

    expect(global.fetch).toHaveBeenCalledWith('/api/ragflow/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: 'user1' }),
    });

    expect(conversationId).toBe('12345');
  });

  test('should send a message', async () => {
    global.fetch.mockResolvedValueOnce({
      json: () => Promise.resolve({ response: 'Hello, user!' }),
    });

    const response = await ragflowService.sendMessage('12345', 'Hi there');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/ragflow/conversations/12345/messages',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'Hi there' }),
      }
    );

    expect(response).toBe('Hello, user!');
  });

  test('should get chat history', async () => {
    const mockHistory = [
      { content: 'Hello', sender: 'user' },
      { content: 'Hi there!', sender: 'assistant' }
    ];

    global.fetch.mockResolvedValueOnce({
      json: () => Promise.resolve(mockHistory),
    });

    const history = await ragflowService.getChatHistory('12345');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/ragflow/conversations/12345/history',
      {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      }
    );

    expect(history).toEqual(mockHistory);
  });
});
