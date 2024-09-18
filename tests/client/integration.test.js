import { App } from '../../public/js/app.js';
import { GraphSimulation } from '../../public/js/components/graphSimulation';
import { ChatManager } from '../../public/js/components/chatManager';
import { Interface } from '../../public/js/components/interface';
import { Visualization } from '../../public/js/components/visualization';
import { WebXRVisualization } from '../../public/js/components/webXRVisualization';
import { GraphDataManager } from '../../public/js/services/graphDataManager';
import { RAGflowService } from '../../public/js/services/ragflowService';
import { WebsocketService } from '../../public/js/services/websocketService';

jest.mock('../../public/js/components/graphSimulation');
jest.mock('../../public/js/components/chatManager');
jest.mock('../../public/js/components/interface');
jest.mock('../../public/js/components/visualization');
jest.mock('../../public/js/components/webXRVisualization');
jest.mock('../../public/js/services/graphDataManager');
jest.mock('../../public/js/services/ragflowService');
jest.mock('../../public/js/services/websocketService');

describe('App Integration', () => {
  let app;

  beforeEach(() => {
    app = new App();
  });

  test('App initializes all components', () => {
    expect(app.websocketService).toBeInstanceOf(WebsocketService);
    expect(app.graphDataManager).toBeInstanceOf(GraphDataManager);
    expect(app.graphSimulation).toBeInstanceOf(GraphSimulation);
    expect(app.visualization).toBeInstanceOf(Visualization);
    expect(app.webXRVisualization).toBeInstanceOf(WebXRVisualization);
    expect(app.interface).toBeInstanceOf(Interface);
    expect(app.chatManager).toBeInstanceOf(ChatManager);
    expect(app.ragflowService).toBeInstanceOf(RAGflowService);
  });

  test('App.start() loads initial data and sets up the application', async () => {
    const mockData = {
      nodes: [{ id: 1, name: 'Node 1' }, { id: 2, name: 'Node 2' }],
      edges: [{ source: 1, target: 2 }]
    };
    app.graphDataManager.loadInitialData = jest.fn().mockResolvedValue(mockData);
    app.graphSimulation.updateNodeData = jest.fn();
    app.graphSimulation.updateEdgeData = jest.fn();
    app.visualization.createNodeObjects = jest.fn();
    app.visualization.createEdgeObjects = jest.fn();
    app.interface.initKeyboardNavigation = jest.fn();

    await app.start();

    expect(app.graphDataManager.loadInitialData).toHaveBeenCalled();
    expect(app.graphSimulation.updateNodeData).toHaveBeenCalledWith(mockData.nodes);
    expect(app.graphSimulation.updateEdgeData).toHaveBeenCalledWith(mockData.edges);
    expect(app.visualization.createNodeObjects).toHaveBeenCalledWith(mockData.nodes);
    expect(app.visualization.createEdgeObjects).toHaveBeenCalledWith(mockData.edges);
    expect(app.interface.initKeyboardNavigation).toHaveBeenCalledWith(app.graphSimulation);
  });

  test('App handles errors when loading initial data', async () => {
    app.graphDataManager.loadInitialData = jest.fn().mockRejectedValue(new Error('Failed to load data'));
    app.interface.displayErrorMessage = jest.fn();

    await app.start();

    expect(app.interface.displayErrorMessage).toHaveBeenCalledWith('Failed to load initial data');
  });

  test('App.update() updates graph simulation and visualization', () => {
    const mockNodePositions = [{ x: 1, y: 2, z: 3 }];
    app.graphSimulation.compute = jest.fn();
    app.graphSimulation.getNodePositions = jest.fn().mockReturnValue(mockNodePositions);
    app.visualization.updateNodePositions = jest.fn();
    app.visualization.updateEdgePositions = jest.fn();

    app.update();

    expect(app.graphSimulation.compute).toHaveBeenCalledWith(expect.any(Number));
    expect(app.graphSimulation.getNodePositions).toHaveBeenCalled();
    expect(app.visualization.updateNodePositions).toHaveBeenCalledWith(mockNodePositions);
    expect(app.visualization.updateEdgePositions).toHaveBeenCalledWith(mockNodePositions);
  });

  test('App.render() calls visualization.animate()', () => {
    app.visualization.animate = jest.fn();

    app.render();

    expect(app.visualization.animate).toHaveBeenCalled();
  });

  test('App handles window resize', () => {
    app.visualization.onWindowResize = jest.fn();

    app.onWindowResize();

    expect(app.visualization.onWindowResize).toHaveBeenCalled();
  });
});