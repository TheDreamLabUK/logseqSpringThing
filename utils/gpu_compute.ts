import { Logger } from './logger';
import forceComputeShader from '../shaders/force_compute.wgsl';
import { MetricsCollector } from './metrics';
import { BufferPool } from './buffer_pool';

const log = new Logger('GPUCompute');

interface SimulationParams {
    springK: number;
    repulsion: number;
    damping: number;
    deltaTime: number;
}

export class GPUCompute {
    private device: GPUDevice | null = null;
    private computePipeline: GPUComputePipeline | null = null;
    private nodeBufferA: GPUBuffer | null = null;
    private nodeBufferB: GPUBuffer | null = null;
    private edgeBuffer: GPUBuffer | null = null;
    private paramsBuffer: GPUBuffer | null = null;
    private bindGroups: GPUBindGroup[] = [];
    private metrics: MetricsCollector;
    private bufferPool: BufferPool;

    private constructor(device: GPUDevice) {
        this.device = device;
        this.metrics = MetricsCollector.getInstance();
        this.bufferPool = new BufferPool();
        this.initializeCompute();
    }

    private async initializeCompute(): Promise<void> {
        if (!this.device) return;

        const shaderModule = this.device.createShaderModule({
            code: forceComputeShader,
        });

        this.computePipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

        // Create uniform buffer for simulation parameters
        this.paramsBuffer = this.device.createBuffer({
            size: 4 * 4, // 4 f32 values
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Set default simulation parameters
        this.updateSimulationParams({
            springK: 0.1,
            repulsion: 1000,
            damping: 0.8,
            deltaTime: 0.016,
        });
    }

    public async setGraphData(nodes: Map<string, Node>, edges: Edge[]): Promise<void> {
        try {
            this.validateDevice();

            if (!this.device) return;

            const nodeArray = Array.from(nodes.values());
            const nodeData = new Float32Array(nodeArray.length * 8); // position(3) + velocity(3) + mass(1) + padding(1)
            
            nodeArray.forEach((node, i) => {
                const offset = i * 8;
                nodeData.set(node.position, offset);
                nodeData.set(node.velocity, offset + 3);
                nodeData[offset + 6] = node.mass;
                nodeData[offset + 7] = 0; // padding
            });

            const edgeData = new Float32Array(edges.length * 4); // source(1) + target(1) + strength(1) + padding(1)
            edges.forEach((edge, i) => {
                const offset = i * 4;
                edgeData[offset] = parseInt(edge.source);
                edgeData[offset + 1] = parseInt(edge.target);
                edgeData[offset + 2] = edge.strength;
                edgeData[offset + 3] = 0; // padding
            });

            // Create node buffers for double buffering
            this.nodeBufferA = this.createBuffer(nodeData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
            this.nodeBufferB = this.createBuffer(nodeData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

            // Create edge buffer
            this.edgeBuffer = this.createBuffer(edgeData, GPUBufferUsage.STORAGE);

            // Create bind groups for double buffering
            this.bindGroups = [
                this.createBindGroup(this.nodeBufferA, this.nodeBufferB),
                this.createBindGroup(this.nodeBufferB, this.nodeBufferA),
            ];
        } catch (error) {
            log.error('Failed to set graph data:', error);
            throw error;
        }
    }

    private createBuffer(data: Float32Array, usage: number): GPUBuffer {
        const buffer = this.device!.createBuffer({
            size: data.byteLength,
            usage,
            mappedAtCreation: true,
        });

        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    private createBindGroup(inputBuffer: GPUBuffer, outputBuffer: GPUBuffer): GPUBindGroup {
        return this.device!.createBindGroup({
            layout: this.computePipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: this.edgeBuffer! } },
                { binding: 2, resource: { buffer: outputBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer! } },
            ],
        });
    }

    public updateSimulationParams(params: SimulationParams): void {
        if (!this.device || !this.paramsBuffer) return;

        const data = new Float32Array([
            params.springK,
            params.repulsion,
            params.damping,
            params.deltaTime,
        ]);

        this.device.queue.writeBuffer(this.paramsBuffer, 0, data);
    }

    private currentBindGroupIndex = 0;

    public async computeForces(): Promise<Map<string, Float32Array>> {
        try {
            this.validateDevice();
            const startTime = performance.now();

            if (!this.device || !this.computePipeline || this.bindGroups.length === 0) {
                throw new Error('GPU compute not initialized');
            }

            const commandEncoder = this.device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();

            computePass.setPipeline(this.computePipeline);
            computePass.setBindGroup(0, this.bindGroups[this.currentBindGroupIndex]);

            const workgroupSize = 256;
            const numWorkgroups = Math.ceil(this.getNodeCount() / workgroupSize);
            computePass.dispatchWorkgroups(numWorkgroups);
            computePass.end();

            this.device.queue.submit([commandEncoder.finish()]);

            // Swap bind groups for next frame
            this.currentBindGroupIndex = 1 - this.currentBindGroupIndex;

            // Read back results
            const endTime = performance.now();
            await this.metrics.recordGpuComputeTime(endTime - startTime);

            return this.readbackResults();
        } catch (error) {
            log.error('Failed to compute forces:', error);
            throw error;
        }
    }

    private async readbackResults(): Promise<Map<string, Float32Array>> {
        if (!this.device || !this.nodeBufferA || !this.nodeBufferB) {
            throw new Error('GPU compute not initialized');
        }

        const size = this.getNodeCount() * 8 * 4; // 8 floats per node
        const stagingBuffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Copy the current output buffer to the staging buffer
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.currentBindGroupIndex === 0 ? this.nodeBufferB : this.nodeBufferA,
            0,
            stagingBuffer,
            0,
            stagingBuffer.size
        );

        this.device.queue.submit([commandEncoder.finish()]);

        // Map the staging buffer and read the results
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const results = new Float32Array(stagingBuffer.getMappedRange());

        // Convert results to Map<string, Float32Array>
        const nodePositions = new Map<string, Float32Array>();
        for (let i = 0; i < this.getNodeCount(); i++) {
            const offset = i * 8;
            const position = new Float32Array(3);
            position.set(results.subarray(offset, offset + 3));
            nodePositions.set(i.toString(), position);
        }

        // Clean up
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        // Get a buffer from the pool for results
        const resultsBuffer = await this.bufferPool.getBuffer(size);
        const results = new Float32Array(resultsBuffer.buffer);

        // Return the buffer to the pool
        await this.bufferPool.returnBuffer(resultsBuffer);

        return nodePositions;
    }

    private getNodeCount(): number {
        if (!this.nodeBufferA) return 0;
        return this.nodeBufferA.size / (8 * 4); // 8 floats per node
    }

    public dispose(): void {
        this.nodeBufferA?.destroy();
        this.nodeBufferB?.destroy();
        this.edgeBuffer?.destroy();
        this.paramsBuffer?.destroy();
        this.device = null;
        this.computePipeline = null;
        this.bindGroups = [];
        this.bufferPool.dispose();
    }

    public static async create(): Promise<GPUCompute | null> {
        try {
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported');
            }

            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            
            if (!adapter) {
                throw new Error('No GPU adapter found');
            }

            const device = await adapter.requestDevice({
                requiredFeatures: ['shader-f32'],
                requiredLimits: {
                    maxStorageBufferBindingSize: 1024 * 1024 * 128, // 128MB
                    maxComputeWorkgroupsPerDimension: 65535,
                },
            });

            return new GPUCompute(device);
        } catch (error) {
            log.error('Failed to initialize GPU compute:', error);
            return null;
        }
    }

    private validateDevice(): void {
        if (!this.device) {
            throw new Error('GPU device not initialized');
        }

        if (!this.device.features.has('shader-f32')) {
            throw new Error('Required feature "shader-f32" not supported');
        }
    }
} 