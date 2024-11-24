// public/js/gpuUtils.js

/**
 * GPU Accelerated Utilities using WebGL Compute Shaders
 */

// WebGL compute shader for force-directed layout
const forceDirectedComputeShader = `#version 310 es
layout(local_size_x = 256) in;

struct Node {
    vec4 position;  // w component used for mass
    vec4 velocity;  // w component used for charge
};

layout(std430, binding = 0) buffer NodesBuffer {
    Node nodes[];
};

layout(std430, binding = 1) buffer LinksBuffer {
    ivec2 links[];  // source and target indices
};

uniform float deltaTime;
uniform float springLength;
uniform float springStiffness;
uniform float repulsion;
uniform float damping;

void main() {
    uint nodeIndex = gl_GlobalInvocationID.x;
    if (nodeIndex >= nodes.length()) return;

    vec3 force = vec3(0.0);
    vec3 pos = nodes[nodeIndex].position.xyz;
    float charge = nodes[nodeIndex].velocity.w;

    // Repulsion forces (node-node)
    for (uint i = 0; i < nodes.length(); i++) {
        if (i == nodeIndex) continue;
        vec3 diff = pos - nodes[i].position.xyz;
        float dist = length(diff);
        if (dist > 0.0) {
            force += normalize(diff) * repulsion * charge * nodes[i].velocity.w / (dist * dist);
        }
    }

    // Spring forces (links)
    for (uint i = 0; i < links.length(); i++) {
        if (links[i].x == int(nodeIndex)) {
            vec3 diff = pos - nodes[links[i].y].position.xyz;
            float dist = length(diff);
            force -= normalize(diff) * springStiffness * (dist - springLength);
        }
        else if (links[i].y == int(nodeIndex)) {
            vec3 diff = pos - nodes[links[i].x].position.xyz;
            float dist = length(diff);
            force -= normalize(diff) * springStiffness * (dist - springLength);
        }
    }

    // Update velocity and position
    vec3 velocity = nodes[nodeIndex].velocity.xyz;
    velocity = (velocity + force * deltaTime) * damping;
    pos += velocity * deltaTime;

    // Write back results
    nodes[nodeIndex].position.xyz = pos;
    nodes[nodeIndex].velocity.xyz = velocity;
}`;

export class GPUAccelerator {
    constructor() {
        this.gl = null;
        this.computeProgram = null;
        this.nodesBuffer = null;
        this.linksBuffer = null;
        this.initialized = false;
    }

    /**
     * Initialize WebGL compute resources
     * @returns {boolean} Success status
     */
    async init() {
        try {
            // Create WebGL 2.0 Compute context
            const canvas = document.createElement('canvas');
            this.gl = canvas.getContext('webgl2-compute');
            
            if (!this.gl) {
                console.warn('WebGL 2.0 Compute not available, falling back to CPU computation');
                return false;
            }

            // Create and compile compute shader
            const shader = this.gl.createShader(this.gl.COMPUTE_SHADER);
            this.gl.shaderSource(shader, forceDirectedComputeShader);
            this.gl.compileShader(shader);

            if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
                console.error('Compute shader compilation failed:', this.gl.getShaderInfoLog(shader));
                return false;
            }

            // Create compute program
            this.computeProgram = this.gl.createProgram();
            this.gl.attachShader(this.computeProgram, shader);
            this.gl.linkProgram(this.computeProgram);

            if (!this.gl.getProgramParameter(this.computeProgram, this.gl.LINK_STATUS)) {
                console.error('Compute program linking failed:', this.gl.getProgramInfoLog(this.computeProgram));
                return false;
            }

            this.initialized = true;
            return true;
        } catch (error) {
            console.error('GPU initialization failed:', error);
            return false;
        }
    }

    /**
     * Allocate GPU buffers for nodes and links
     * @param {number} numNodes - Number of nodes
     * @param {number} numLinks - Number of links
     */
    allocateBuffers(numNodes, numLinks) {
        if (!this.initialized) return;

        // Create and bind nodes buffer
        this.nodesBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.nodesBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, numNodes * 32, this.gl.DYNAMIC_DRAW); // 32 bytes per node (2 vec4s)
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 0, this.nodesBuffer);

        // Create and bind links buffer
        this.linksBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.linksBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, numLinks * 8, this.gl.DYNAMIC_DRAW); // 8 bytes per link (2 ints)
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 1, this.linksBuffer);
    }

    /**
     * Update node data in GPU buffer
     * @param {Float32Array} nodeData - Node positions and properties
     */
    updateNodes(nodeData) {
        if (!this.initialized) return;
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.nodesBuffer);
        this.gl.bufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, nodeData);
    }

    /**
     * Update link data in GPU buffer
     * @param {Int32Array} linkData - Link source and target indices
     */
    updateLinks(linkData) {
        if (!this.initialized) return;
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.linksBuffer);
        this.gl.bufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, linkData);
    }

    /**
     * Compute force-directed layout on GPU
     * @param {object} params - Layout parameters
     * @returns {Float32Array} Updated node positions
     */
    computeLayout(params) {
        if (!this.initialized) return null;

        try {
            // Use compute program
            this.gl.useProgram(this.computeProgram);

            // Set uniforms
            const uniforms = {
                deltaTime: params.deltaTime || 0.016,
                springLength: params.springLength || 1.0,
                springStiffness: params.springStiffness || 0.1,
                repulsion: params.repulsion || 1.0,
                damping: params.damping || 0.98
            };

            for (const [name, value] of Object.entries(uniforms)) {
                const location = this.gl.getUniformLocation(this.computeProgram, name);
                this.gl.uniform1f(location, value);
            }

            // Dispatch compute shader
            const workGroupSize = 256;
            const numWorkGroups = Math.ceil(params.numNodes / workGroupSize);
            this.gl.dispatchCompute(numWorkGroups, 1, 1);

            // Wait for computation to complete
            this.gl.memoryBarrier(this.gl.SHADER_STORAGE_BARRIER_BIT);

            // Read back results
            const resultBuffer = new Float32Array(params.numNodes * 8); // 8 floats per node
            this.gl.getBufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, resultBuffer);

            return resultBuffer;
        } catch (error) {
            console.error('GPU computation failed:', error);
            return null;
        }
    }

    /**
     * Clean up GPU resources
     */
    dispose() {
        if (!this.initialized) return;

        try {
            // Delete buffers
            if (this.nodesBuffer) this.gl.deleteBuffer(this.nodesBuffer);
            if (this.linksBuffer) this.gl.deleteBuffer(this.linksBuffer);

            // Delete program and shader
            if (this.computeProgram) {
                const shader = this.gl.getAttachedShaders(this.computeProgram)[0];
                this.gl.deleteShader(shader);
                this.gl.deleteProgram(this.computeProgram);
            }

            this.initialized = false;
        } catch (error) {
            console.error('Error disposing GPU resources:', error);
        }
    }
}

// Export singleton instance
export const gpuAccelerator = new GPUAccelerator();

/**
 * Check if GPU acceleration is available
 * @returns {Promise<boolean>} True if GPU is available
 */
export async function isGPUAvailable() {
    try {
        return await gpuAccelerator.init();
    } catch (e) {
        console.error('GPU availability check failed:', e);
        return false;
    }
}

/**
 * Initialize GPU computation utilities
 * @returns {Promise<GPUAccelerator>} GPU accelerator instance
 */
export async function initGPU() {
    if (await isGPUAvailable()) {
        return gpuAccelerator;
    }
    return null;
}

/**
 * Perform computations on the GPU
 * @param {GPUAccelerator} gpu - GPU accelerator instance
 * @param {object} data - Computation data
 * @param {object} params - Computation parameters
 * @returns {Float32Array|null} Computation results
 */
export function computeOnGPU(gpu, data, params) {
    if (!gpu?.initialized) return null;

    try {
        // Prepare node and link data
        const nodeData = new Float32Array(data.nodes.flatMap(node => [
            node.x, node.y, node.z, node.mass || 1.0,
            node.vx || 0, node.vy || 0, node.vz || 0, node.charge || 1.0
        ]));

        const linkData = new Int32Array(data.links.flatMap(link => [
            link.source, link.target
        ]));

        // Allocate buffers if needed
        gpu.allocateBuffers(data.nodes.length, data.links.length);

        // Update data
        gpu.updateNodes(nodeData);
        gpu.updateLinks(linkData);

        // Compute layout
        return gpu.computeLayout({
            ...params,
            numNodes: data.nodes.length
        });
    } catch (error) {
        console.error('Error in GPU computation:', error);
        return null;
    }
}
