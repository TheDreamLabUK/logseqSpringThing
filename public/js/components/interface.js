export class Interface {
  constructor(document) {
    this.document = document;
  }

  createUI() {
    const infoPanel = this.document.createElement('div');
    infoPanel.id = 'node-info-panel';
    infoPanel.style.position = 'absolute';
    infoPanel.style.top = '10px';
    infoPanel.style.left = '10px';
    infoPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    infoPanel.style.color = 'white';
    infoPanel.style.padding = '10px';
    infoPanel.style.borderRadius = '5px';
    infoPanel.style.display = 'none';
    this.document.body.appendChild(infoPanel);
  }

  updateNodeInfoUI(node) {
    const infoPanel = this.document.getElementById('node-info-panel');
    if (node) {
      infoPanel.innerHTML = `
        <h3>Node Information</h3>
        <p>ID: ${node.id}</p>
        <p>Name: ${node.name}</p>
        <p>Size: ${node.size || 'N/A'}</p>
      `;
      infoPanel.style.display = 'block';
    } else {
      infoPanel.style.display = 'none';
    }
  }

  createChatInterface() {
    const chatContainer = this.document.createElement('div');
    chatContainer.id = 'chat-container';
    chatContainer.style.position = 'absolute';
    chatContainer.style.bottom = '10px';
    chatContainer.style.right = '10px';
    chatContainer.style.width = '300px';
    chatContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    chatContainer.style.color = 'white';
    chatContainer.style.padding = '10px';
    chatContainer.style.borderRadius = '5px';

    const chatMessages = this.document.createElement('div');
    chatMessages.id = 'chat-messages';
    chatMessages.style.height = '200px';
    chatMessages.style.overflowY = 'auto';
    chatMessages.style.marginBottom = '10px';

    const chatInput = this.document.createElement('input');
    chatInput.type = 'text';
    chatInput.id = 'chat-input';
    chatInput.placeholder = 'Ask a question...';
    chatInput.style.width = '100%';
    chatInput.style.padding = '5px';

    chatContainer.appendChild(chatMessages);
    chatContainer.appendChild(chatInput);
    this.document.body.appendChild(chatContainer);
  }

  addChatMessage(sender, message) {
    const chatMessages = this.document.getElementById('chat-messages');
    const messageElement = this.document.createElement('p');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  displayErrorMessage(message) {
    const errorContainer = this.document.createElement('div');
    errorContainer.style.position = 'fixed';
    errorContainer.style.top = '50%';
    errorContainer.style.left = '50%';
    errorContainer.style.transform = 'translate(-50%, -50%)';
    errorContainer.style.backgroundColor = 'rgba(255, 0, 0, 0.8)';
    errorContainer.style.color = 'white';
    errorContainer.style.padding = '20px';
    errorContainer.style.borderRadius = '5px';
    errorContainer.style.zIndex = '1000';
    errorContainer.textContent = message;

    this.document.body.appendChild(errorContainer);

    setTimeout(() => {
      this.document.body.removeChild(errorContainer);
    }, 5000);
  }
}
