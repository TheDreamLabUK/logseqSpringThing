import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js';

export class GraphSimulation {
    constructor(nodes, edges, renderer) {
        this.nodes = nodes;
        this.edges = edges;
        this.renderer = renderer;
        this.initGPUComputation();
    }

    initGPUComputation() {
        const width = Math.ceil(Math.sqrt(this.nodes.length));
        const height = Math.ceil(this.nodes.length / width);

        this.gpuCompute = new GPUComputationRenderer(width, height, this.renderer);

        const positionTexture = this.gpuCompute.createTexture();
        const velocityTexture = this.gpuCompute.createTexture();

        this.fillPositionTexture(positionTexture);
        this.fillVelocityTexture(velocityTexture);

        this.positionVariable = this.gpuCompute.addVariable('texturePosition', this.getPositionShader(), positionTexture);
        this.velocityVariable = this.gpuCompute.addVariable('textureVelocity', this.getVelocityShader(), velocityTexture);

        this.gpuCompute.setVariableDependencies(this.positionVariable, [this.positionVariable, this.velocityVariable]);
        this.gpuCompute.setVariableDependencies(this.velocityVariable, [this.positionVariable, this.velocityVariable]);

        const error = this.gpuCompute.init();
        if (error !== null) {
            console.error('GPUComputationRenderer init error:', error);
        }
    }

    fillPositionTexture(texture) {
        const theArray = texture.image.data;
        for (let i = 0; i < this.nodes.length; i++) {
            const stride = i * 4;
            theArray[stride] = this.nodes[i].x;
            theArray[stride + 1] = this.nodes[i].y;
            theArray[stride + 2] = this.nodes[i].z;
            theArray[stride + 3] = 1;
        }
    }

    fillVelocityTexture(texture) {
        const theArray = texture.image.data;
        for (let i = 0; i < theArray.length; i++) {
            theArray[i] = 0;
        }
    }

    getPositionShader() {
        return `
            uniform float delta;
            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec4 pos = texture2D(texturePosition, uv);
                vec4 vel = texture2D(textureVelocity, uv);
                pos.xyz += vel.xyz * delta;
                gl_FragColor = pos;
            }
        `;
    }

    getVelocityShader() {
        return `
            uniform float delta;
            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec4 pos = texture2D(texturePosition, uv);
                vec4 vel = texture2D(textureVelocity, uv);
                
                // Simple force calculation (can be expanded for more complex simulations)
                vec3 force = vec3(0.0);
                for (float y = 0.0; y < 1.0; y += 1.0 / resolution.y) {
                    for (float x = 0.0; x < 1.0; x += 1.0 / resolution.x) {
                        vec3 otherPos = texture2D(texturePosition, vec2(x, y)).xyz;
                        vec3 diff = pos.xyz - otherPos;
                        float dist = length(diff);
                        if (dist > 0.0 && dist < 50.0) {
                            force += normalize(diff) / (dist * dist);
                        }
                    }
                }
                
                vel.xyz = vel.xyz + force * delta;
                vel.xyz *= 0.99; // damping
                
                gl_FragColor = vel;
            }
        `;
    }

    compute(deltaTime) {
        this.gpuCompute.compute();
    }

    getNodePositions() {
        return this.gpuCompute.getCurrentRenderTarget(this.positionVariable).texture;
    }

    updatePositions(updates) {
        // This method is kept for compatibility, but won't be used with GPU acceleration
        console.warn('updatePositions called, but GPU acceleration is in use');
    }

    updateNodeData(newNodes) {
        this.nodes = newNodes;
        this.initGPUComputation();
    }

    updateEdgeData(newEdges) {
        this.edges = newEdges;
        // Note: For this simple implementation, edges don't affect the GPU computation
    }
}
