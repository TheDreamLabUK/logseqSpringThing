// public/js/app.js

import { ChatManager } from './components/chatManager.js';
import { Interface } from './components/interface.js';
import { Visualization } from './components/visualization.js';
import { WebXRVisualization } from './components/webXRVisualization.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { RAGflowService } from './services/ragflowService.js';
import { WebsocketService } from './services/websocketService.js';

/**
 * Main application class that initializes all components and manages the application lifecycle.
 */
export class App {
  constructor() {
    // Initialize all necessary components
    this.initializeComponents();
  }

  /**
   * Initializes all components required for the application.
   */
  initializeComponents() {
    // Initialize WebSocket service with the server's WebSocket URL
    this.websocketService = new WebsocketService('wss://localhost:8443'); // Update URL as per your server configuration

    // Initialize GraphDataManager with the WebSocket service
    this.graphDataManager = new GraphDataManager(this.websocketService);

    // Initialize Visualization component for rendering the graph
    this.visualization = new Visualization();

    // Initialize WebXRVisualization for immersive VR/AR experiences
    this.webXRVisualization = new WebXRVisualization(
      this.visualization.scene,
      this.visualization.renderer,
      this.visualization.camera,
      this.websocketService
    );

    // Initialize Interface component for UI interactions and error handling
    this.interface = new Interface(document);

    // Initialize ChatManager for handling chat functionalities
    this.chatManager = new ChatManager(this.websocketService);

    // Initialize RAGFlowService for AI-powered question answering
    this.ragflowService = new RAGflowService(this.websocketService);
  }

  /**
   * Starts the application by loading initial data, setting up event listeners, and initiating the animation loop.
   */
  async start() {
    // Load initial graph data from the server
    await this.loadInitialData();

    // Set up necessary event listeners
    this.setupEventListeners();

    // Start the rendering animation loop
    this.startAnimationLoop();
  }

  /**
   * Sets up event listeners for window resizing and other interactions.
   */
  setupEventListeners() {
    // Handle window resize events to adjust the visualization accordingly
    window.addEventListener('resize', this.onWindowResize.bind(this), false);

    // Initialize keyboard navigation or other UI interactions if needed
    // Example: this.interface.initKeyboardNavigation(this.graphSimulation);
    // Note: Since simulation is now server-side, this line may need adjustments or removal
  }

  /**
   * Loads initial graph data from the server and sets up real-time updates via WebSocket.
   */
  async loadInitialData() {
    try {
      // Fetch initial graph data from the server
      await this.graphDataManager.fetchInitialData();

      // Create the initial graph visualization based on fetched data
      this.visualization.createGraph(this.graphDataManager.graphData);

      // Subscribe to real-time graph updates via WebSocket
      this.graphDataManager.subscribeToUpdates((updatedData) => {
        this.visualization.updateGraph(updatedData);
      });

      // Optionally, initialize chat functionalities
      this.chatManager.initializeChat();
    } catch (error) {
      // Display an error message if initial data loading fails
      this.interface.displayErrorMessage('Failed to load initial data');
    }
  }

  /**
   * Starts the animation loop to render the visualization continuously.
   */
  startAnimationLoop() {
    const animate = () => {
      requestAnimationFrame(animate);
      this.render();
    };
    animate();
  }

  /**
   * Renders the visualization using Three.js.
   */
  render() {
    this.visualization.render();
  }

  /**
   * Handles window resize events by updating the visualization dimensions.
   */
  onWindowResize() {
    this.visualization.onWindowResize();
  }
}
