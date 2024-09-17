// Mock RAGFlowService class
class RAGFlowService {
  async createConversation(userId) {
    return 'mock-conversation-id';
  }

  async sendMessage(conversationId, message) {
    return 'mock-response';
  }

  async getChatHistory(conversationId) {
    return [
      { content: 'Hello', sender: 'user' },
      { content: 'Hi there!', sender: 'assistant' }
    ];
  }
}

describe('RAGFlowService', () => {
  let ragflowService;

  beforeEach(() => {
    ragflowService = new RAGFlowService();
  });

  test('createConversation should return a conversation ID', async () => {
    const userId = 'testUser';
    const conversationId = await ragflowService.createConversation(userId);
    expect(conversationId).toBeDefined();
    expect(typeof conversationId).toBe('string');
  });

  test('sendMessage should return a response', async () => {
    const conversationId = 'testConversation';
    const message = 'Hello, RAGFlow!';
    const response = await ragflowService.sendMessage(conversationId, message);
    expect(response).toBeDefined();
    expect(typeof response).toBe('string');
  });

  test('getChatHistory should return an array of messages', async () => {
    const conversationId = 'testConversation';
    const history = await ragflowService.getChatHistory(conversationId);
    expect(Array.isArray(history)).toBe(true);
    history.forEach(message => {
      expect(message).toHaveProperty('content');
      expect(message).toHaveProperty('sender');
    });
  });

  // Add more tests as needed based on RAGFlowService functionality
});



these are updated?

// ragflowService.test.js

const RagflowService = require('../../public/js/services/ragflowService');

describe('RAGFlowService', () => {
  let ragflowService;

  beforeEach(() => {
    ragflowService = new RagflowService();
  });

  afterEach(() => {
    global.fetch.mockClear();
    delete global.fetch;
  });

  test('should initialize properly', () => {
    expect(ragflowService).toBeDefined();
  });

  test('should create a new conversation', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ conversationId: '12345' }),
      })
    );

    const conversationId = await ragflowService.createConversation('user1');

    expect(global.fetch).toHaveBeenCalledWith('/api/ragflow/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: 'user1' }),
    });

    expect(conversationId).toBe('12345');
  });

  test('should send a message', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ response: 'Hello, user!' }),
      })
    );

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
});
