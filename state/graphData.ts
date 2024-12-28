import { Logger } from '../utils/logger';
import { BinaryProtocol, MessageType } from '../websocket/binaryProtocol';
import { BINARY_PROTOCOL } from '../websocket/constants';
import { GraphCache } from '../utils/graph_cache';
import { GPUCompute } from '../utils/gpu_compute';

const log = new Logger('GraphDataManager');

export interface Node {
    id: string;
    position: Float32Array; // [x, y, z]
    velocity: Float32Array; // [vx, vy, vz]
    mass: number;
}

export interface Edge {
    source: string;
    target: string;
    strength: number;
}

export class GraphDataManager {
    private static instance: GraphDataManager | null = null;
    private nodes: Map<string, Node> = new Map();
    private edges: Edge[] = [];
    private gpuCompute: GPUCompute | null = null;

    private constructor() {
        this.initializeGPUCompute();
    }

    public static getInstance(): GraphDataManager {
        if (!GraphDataManager.instance) {
            GraphDataManager.instance = new GraphDataManager();
        }
        return GraphDataManager.instance;
    }

    private async initializeGPUCompute(): Promise<void> {
        try {
            // Only initialize if WebGPU is supported
            if (!navigator.gpu) {
                log.warn('WebGPU not supported, falling back to CPU');
                this.gpuCompute = null;
                return;
            }

            this.gpuCompute = await GPUCompute.create();
            log.info('GPU compute initialized');
        } catch (error) {
            log.warn('Failed to initialize GPU compute, falling back to CPU:', error);
            this.gpuCompute = null;
        }
    }

    public async loadInitialGraphData(): Promise<void> {
        try {
            // Try to load from cache first
            const cachedData = await GraphCache.loadFromCache();
            if (cachedData) {
                this.setGraphData(cachedData);
                log.info('Graph data loaded from cache');
                return;
            }

            const response = await fetch('/api/graph/data');
            if (!response.ok) {
                throw new Error(`Failed to load graph data: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.setGraphData(data);
            
            // Save to cache
            await GraphCache.saveToCache(
                Array.from(this.nodes.values()),
                this.edges
            );

            if (this.gpuCompute) {
                await this.gpuCompute.setGraphData(this.nodes, this.edges);
            }
        } catch (error) {
            log.error('Failed to load initial graph data:', error);
            throw error;
        }
    }

    private setGraphData(data: { nodes: any[], edges: Edge[] }): void {
        this.nodes.clear();
        data.nodes.forEach(node => {
            this.nodes.set(node.id, {
                ...node,
                position: new Float32Array(3),
                velocity: new Float32Array(3)
            });
        });
        this.edges = data.edges;
    }

    public async updatePositions(): Promise<Float32Array> {
        const positions = new Float32Array(this.nodes.size * 3);
        let offset = 0;

        if (this.gpuCompute) {
            // Use GPU for force computation
            const updates = await this.gpuCompute.computeForces();
            for (const [id, position] of updates) {
                const node = this.nodes.get(id);
                if (node) {
                    node.position.set(position);
                    positions.set(position, offset);
                    offset += 3;
                }
            }
        } else {
            // Fallback to CPU computation
            this.computeForcesOnCPU();
            for (const node of this.nodes.values()) {
                positions.set(node.position, offset);
                offset += 3;
            }
        }

        return positions;
    }

    private computeForcesOnCPU(): void {
        // Simple force-directed layout implementation
        const k = 0.1; // Spring constant
        const repulsion = 1000; // Repulsion constant
        const damping = 0.8; // Velocity damping

        // Calculate forces
        for (const node1 of this.nodes.values()) {
            let fx = 0, fy = 0, fz = 0;

            // Repulsion between all nodes
            for (const node2 of this.nodes.values()) {
                if (node1 === node2) continue;

                const dx = node1.position[0] - node2.position[0];
                const dy = node1.position[1] - node2.position[1];
                const dz = node1.position[2] - node2.position[2];
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;

                const force = repulsion / (distance * distance);
                fx += dx * force / distance;
                fy += dy * force / distance;
                fz += dz * force / distance;
            }

            // Spring forces along edges
            for (const edge of this.edges) {
                if (edge.source === node1.id || edge.target === node1.id) {
                    const other = this.nodes.get(edge.source === node1.id ? edge.target : edge.source);
                    if (!other) continue;

                    const dx = node1.position[0] - other.position[0];
                    const dy = node1.position[1] - other.position[1];
                    const dz = node1.position[2] - other.position[2];
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                    const force = -k * distance * edge.strength;
                    fx += dx * force / distance;
                    fy += dy * force / distance;
                    fz += dz * force / distance;
                }
            }

            // Update velocity and position
            node1.velocity[0] = (node1.velocity[0] + fx) * damping;
            node1.velocity[1] = (node1.velocity[1] + fy) * damping;
            node1.velocity[2] = (node1.velocity[2] + fz) * damping;

            node1.position[0] += node1.velocity[0];
            node1.position[1] += node1.velocity[1];
            node1.position[2] += node1.velocity[2];
        }
    }

    public dispose(): void {
        if (this.gpuCompute) {
            this.gpuCompute.dispose();
        }
        this.nodes.clear();
        this.edges = [];
    }

    public isGPUComputeAvailable(): boolean {
        return this.gpuCompute !== null;
    }

    public async updateGPUComputeStatus(enabled: boolean): Promise<void> {
        try {
            const response = await fetch('/api/gpu/status', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(enabled),
            });

            if (!response.ok) {
                throw new Error(`Failed to update GPU compute status: ${response.statusText}`);
            }

            if (enabled && !this.gpuCompute) {
                await this.initializeGPUCompute();
            } else if (!enabled && this.gpuCompute) {
                this.gpuCompute.dispose();
                this.gpuCompute = null;
            }
        } catch (error) {
            log.error('Failed to update GPU compute status:', error);
            throw error;
        }
    }
} 