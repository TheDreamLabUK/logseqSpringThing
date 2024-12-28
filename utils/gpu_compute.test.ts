import { GPUCompute } from './gpu_compute';
import { Node, Edge } from '../state/graphData';

describe('GPUCompute', () => {
    let gpuCompute: GPUCompute;

    beforeAll(async () => {
        try {
            gpuCompute = await GPUCompute.create();
        } catch (error) {
            console.warn('GPU compute not available, skipping tests:', error);
        }
    });

    afterAll(() => {
        gpuCompute?.dispose();
    });

    it('should initialize GPU compute', () => {
        if (!gpuCompute) {
            return;
        }
        expect(gpuCompute).toBeDefined();
    });

    it('should set graph data', async () => {
        if (!gpuCompute) {
            return;
        }

        const nodes = new Map<string, Node>();
        nodes.set('1', {
            id: '1',
            position: new Float32Array([0, 0, 0]),
            velocity: new Float32Array([0, 0, 0]),
            mass: 1,
        });

        const edges: Edge[] = [];

        await expect(gpuCompute.setGraphData(nodes, edges)).resolves.not.toThrow();
    });

    it('should compute forces', async () => {
        if (!gpuCompute) {
            return;
        }

        const nodes = new Map<string, Node>();
        nodes.set('1', {
            id: '1',
            position: new Float32Array([0, 0, 0]),
            velocity: new Float32Array([0, 0, 0]),
            mass: 1,
        });
        nodes.set('2', {
            id: '2',
            position: new Float32Array([1, 0, 0]),
            velocity: new Float32Array([0, 0, 0]),
            mass: 1,
        });

        const edges = [{
            source: '1',
            target: '2',
            strength: 1,
        }];

        await gpuCompute.setGraphData(nodes, edges);
        const results = await gpuCompute.computeForces();

        expect(results.size).toBe(nodes.size);
        for (const position of results.values()) {
            expect(position).toBeInstanceOf(Float32Array);
            expect(position.length).toBe(3);
        }
    });
}); 