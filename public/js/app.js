import { GraphSimulation } from './components/graphSimulation.js';
import { ChatManager } from './components/chatManager.js';
import { Interface } from './components/interface.js';
import { Visualization } from './components/visualization.js';
import { WebXRVisualization } from './components/webXRVisualization.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { RAGflowService } from './services/ragflowService.js';
import { WebsocketService } from './services/websocketService.js';

export class App {
  constructor() {
    this.initializeComponents();
  }

  initializeComponents() {
    this.websocketService = new WebsocketService('wss://example.com');
    this.graphDataManager = new GraphDataManager();
    this.graphSimulation = new GraphSimulation();
    this.visualization = new Visualization();
    this.webXRVisualization = new WebXRVisualization(this.visualization.scene, this.visualization.renderer, this.visualization.camera);
    this.interface = new Interface(document);
    this.chatManager = new ChatManager(this.websocketService);
    this.ragflowService = new RAGflowService(this.websocketService);
  }

  async start() {
    await this.loadInitialData();
    this.setupEventListeners();
    this.startAnimationLoop();
  }

  setupEventListeners() {
    window.addEventListener('resize', this.onWindowResize.bind(this), false);
    this.interface.initKeyboardNavigation(this.graphSimulation);
  }

  async loadInitialData() {
    try {
      const data = await this.graphDataManager.loadInitialData();
      this.graphSimulation.updateNodeData(data.nodes);
      this.graphSimulation.updateEdgeData(data.edges);
      this.visualization.createNodeObjects(data.nodes);
      this.visualization.createEdgeObjects(data.edges);
    } catch (error) {
      this.interface.displayErrorMessage('Failed to load initial data');
    }
  }

  startAnimationLoop() {
    const animate = () => {
      requestAnimationFrame(animate);
      this.update();
      this.render();
    };
    animate();
  }

  update() {
    const deltaTime = 0.016; // Assuming 60fps
    this.graphSimulation.compute(deltaTime);
    const nodePositions = this.graphSimulation.getNodePositions();
    this.visualization.updateNodePositions(nodePositions);
    this.visualization.updateEdgePositions(nodePositions);
  }

  render() {
    this.visualization.animate();
  }

  onWindowResize() {
    this.visualization.onWindowResize();
  }
}