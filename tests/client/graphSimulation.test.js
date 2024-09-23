import { GraphSimulation } from '../../public/js/components/graphSimulation';

jest.mock('three');
jest.mock('three/examples/jsm/misc/GPUComputationRenderer.js');

describe('GraphSimulation', () => {
  let graphSimulation;
  let mockRenderer;
  const mockNodes = [
    { id: 1, name: 'Node 1', x: 0, y: 0, z: 0 },
    { id: 2, name: 'Node 2', x: 1, y: 1, z: 1 },
  ];
  const mockEdges = [
    { source: 0, target: 1 },
  ];

  beforeEach(() => {
    mockRenderer = {
      capabilities: {
        isWebGL2: true
      }
    };
    graphSimulation = new GraphSimulation(mockNodes, mockEdges, mockRenderer);
  });

  test('GraphSimulation constructor initializes correctly', () => {
    expect(graphSimulation.nodes).toEqual(mockNodes);
    expect(graphSimulation.edges).toEqual(mockEdges);
    expect(graphSimulation.renderer).toBe(mockRenderer);
  });

  test('initGPUComputation sets up GPU computation', () => {
    const mockGPUCompute = {
      createTexture: jest.fn(),
      addVariable: jest.fn(),
      setVariableDependencies: jest.fn(),
      init: jest.fn()
    };
    GPUComputationRenderer.mockImplementation(() => mockGPUCompute);

    graphSimulation.initGPUComputation();

    expect(GPUComputationRenderer).toHaveBeenCalled();
    expect(mockGPUCompute.createTexture).toHaveBeenCalledTimes(2);
    expect(mockGPUCompute.addVariable).toHaveBeenCalledTimes(2);
    expect(mockGPUCompute.setVariableDependencies).toHaveBeenCalledTimes(2);
    expect(mockGPUCompute.init).toHaveBeenCalled();
  });

  test('fillPositionTexture fills texture with correct data', () => {
    const mockTexture = {
      image: {
        data: new Float32Array(mockNodes.length * 4)
      }
    };

    graphSimulation.fillPositionTexture(mockTexture);

    expect(mockTexture.image.data[0]).toBe(mockNodes[0].x);
    expect(mockTexture.image.data[1]).toBe(mockNodes[0].y);
    expect(mockTexture.image.data[2]).toBe(mockNodes[0].z);
    expect(mockTexture.image.data[3]).toBe(1);
  });

  test('fillVelocityTexture fills texture with zeros', () => {
    const mockTexture = {
      image: {
        data: new Float32Array(mockNodes.length * 4)
      }
    };

    graphSimulation.fillVelocityTexture(mockTexture);

    expect(mockTexture.image.data.every(val => val === 0)).toBe(true);
  });

  test('getPositionShader returns correct shader code', () => {
    const shader = graphSimulation.getPositionShader();
    expect(shader).toContain('uniform float delta');
    expect(shader).toContain('vec4 pos = texture2D(texturePosition, uv)');
    expect(shader).toContain('vec4 vel = texture2D(textureVelocity, uv)');
    expect(shader).toContain('pos.xyz += vel.xyz * delta');
  });

  test('getVelocityShader returns correct shader code', () => {
    const shader = graphSimulation.getVelocityShader();
    expect(shader).toContain('uniform float delta');
    expect(shader).toContain('vec3 force = vec3(0.0)');
    expect(shader).toContain('force += normalize(diff) / (dist * dist)');
    expect(shader).toContain('vel.xyz *= 0.99');
  });

  test('compute calls gpuCompute.compute', () => {
    const mockCompute = jest.fn();
    graphSimulation.gpuCompute = { compute: mockCompute };

    graphSimulation.compute(0.016);

    expect(mockCompute).toHaveBeenCalled();
  });

  test('getNodePositions returns current node positions', () => {
    const mockRenderTarget = {
      texture: 'mockTexture'
    };
    graphSimulation.positionVariable = { renderTargets: [mockRenderTarget] };
    graphSimulation.gpuCompute = {
      getCurrentRenderTarget: jest.fn().mockReturnValue(mockRenderTarget)
    };

    const result = graphSimulation.getNodePositions();

    expect(result).toBe('mockTexture');
    expect(graphSimulation.gpuCompute.getCurrentRenderTarget).toHaveBeenCalledWith(graphSimulation.positionVariable);
  });

  test('updatePositions logs warning when called', () => {
    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
    graphSimulation.updatePositions([]);
    expect(consoleWarnSpy).toHaveBeenCalledWith('updatePositions called, but GPU acceleration is in use');
    consoleWarnSpy.mockRestore();
  });

  test('updateNodeData updates nodes and reinitializes GPU compute', () => {
    const newNodes = [
      { id: 3, name: 'Node 3' },
      { id: 4, name: 'Node 4' },
    ];
    const initGPUComputationSpy = jest.spyOn(graphSimulation, 'initGPUComputation').mockImplementation();

    graphSimulation.updateNodeData(newNodes);

    expect(graphSimulation.nodes).toEqual(newNodes);
    expect(initGPUComputationSpy).toHaveBeenCalled();
  });

  test('updateEdgeData updates edges', () => {
    const newEdges = [
      { source: 2, target: 3 },
    ];

    graphSimulation.updateEdgeData(newEdges);

    expect(graphSimulation.edges).toEqual(newEdges);
  });

  test('GraphSimulation handles empty graph', () => {
    const emptySimulation = new GraphSimulation([], [], mockRenderer);

    expect(() => {
      emptySimulation.initGPUComputation();
      emptySimulation.compute(0.016);
    }).not.toThrow();
  });

  test('GraphSimulation handles large graph', () => {
    const largeNodes = Array(1000).fill().map((_, i) => ({ id: i, name: `Node ${i}`, x: 0, y: 0, z: 0 }));
    const largeEdges = Array(2000).fill().map((_, i) => ({ source: Math.floor(i/2), target: Math.floor(i/2) + 1 }));
    const largeSimulation = new GraphSimulation(largeNodes, largeEdges, mockRenderer);

    expect(() => {
      largeSimulation.initGPUComputation();
      largeSimulation.compute(0.016);
    }).not.toThrow();
  });
});
