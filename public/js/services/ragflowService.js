export class RagflowService {
  async createConversation(userId) {
    const response = await fetch('/api/ragflow/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId }),
    });
    const data = await response.json();
    return data.conversationId;
  }

  async sendMessage(conversationId, message) {
    const response = await fetch(`/api/ragflow/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    return data.response;
  }

  async getChatHistory(conversationId) {
    const response = await fetch(`/api/ragflow/conversations/${conversationId}/history`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    return await response.json();
  }
}