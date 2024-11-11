// public/js/components/interface.js

/**
 * Interface class manages UI elements like error messages and information panels.
 */
export class Interface {
  /**
   * Creates a new Interface instance.
   * @param {Document} document - The DOM document.
   */
  constructor(document) {
      this.document = document;
      this.createUI();
      this.setupEventListeners();
  }

  /**
   * Creates necessary UI elements and appends them to the DOM.
   */
  createUI() {
      // Create Node Information Panel
      this.createNodeInfoPanel();
  }

  /**
   * Creates a panel to display information about selected nodes.
   */
  createNodeInfoPanel() {
      const infoPanel = this.document.createElement('div');
      infoPanel.id = 'node-info-panel';
      infoPanel.style.position = 'absolute';
      infoPanel.style.top = '20px';
      infoPanel.style.left = '20px';
      infoPanel.style.width = '300px';
      infoPanel.style.maxHeight = '40vh';
      infoPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
      infoPanel.style.color = 'white';
      infoPanel.style.padding = '15px';
      infoPanel.style.borderRadius = '8px';
      infoPanel.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
      infoPanel.style.overflowY = 'auto';
      infoPanel.style.display = 'none'; // Hidden by default
      this.document.body.appendChild(infoPanel);

      this.nodeInfoPanel = infoPanel;
  }

  /**
   * Sets up event listeners for custom events.
   */
  setupEventListeners() {
      // Listen for node selection events
      window.addEventListener('nodeSelected', (event) => {
          const nodeInfo = event.detail;
          this.updateNodeInfoPanel(nodeInfo);
      });
  }

  /**
   * Displays an error message on the screen.
   * @param {string} message - The error message to display.
   */
  displayErrorMessage(message) {
      const errorContainer = this.document.createElement('div');
      errorContainer.style.position = 'fixed';
      errorContainer.style.top = '50%';
      errorContainer.style.left = '50%';
      errorContainer.style.transform = 'translate(-50%, -50%)';
      errorContainer.style.backgroundColor = 'rgba(255, 0, 0, 0.85)';
      errorContainer.style.color = 'white';
      errorContainer.style.padding = '20px';
      errorContainer.style.borderRadius = '8px';
      errorContainer.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.3)';
      errorContainer.style.zIndex = '1000';
      errorContainer.textContent = message;

      this.document.body.appendChild(errorContainer);

      // Remove the error message after 5 seconds
      setTimeout(() => {
          if (this.document.body.contains(errorContainer)) {
              this.document.body.removeChild(errorContainer);
          }
      }, 5000);
  }

  /**
   * Updates the Node Information Panel with details of the selected node.
   * @param {object} node - The node object containing its details.
   */
  updateNodeInfoPanel(node) {
      if (!node) {
          this.nodeInfoPanel.style.display = 'none';
          return;
      }

      this.nodeInfoPanel.innerHTML = `
          <h3>Node Information</h3>
          <p><strong>ID:</strong> ${node.id}</p>
          <p><strong>Name:</strong> ${node.name}</p>
          <!-- Add more node properties as needed -->
      `;
      this.nodeInfoPanel.style.display = 'block';
  }
}
