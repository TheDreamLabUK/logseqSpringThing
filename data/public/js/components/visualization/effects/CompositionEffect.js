import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { LAYERS } from '../layerManager.js';

export class CompositionEffect {
    constructor(renderer) {
        if (!renderer || !renderer.domElement) {
            throw new Error('Invalid renderer provided to CompositionEffect');
        }
        this.renderer = renderer;
        this.composer = null;
        this.initialized = false;
    }

    createRenderTarget() {
        const pixelRatio = this.renderer.getPixelRatio();
        const width = Math.floor(window.innerWidth * pixelRatio);
        const height = Math.floor(window.innerHeight * pixelRatio);

        return new THREE.WebGLRenderTarget(
            width,
            height,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType,
                colorSpace: THREE.SRGBColorSpace,
                stencilBuffer: false,
                depthBuffer: true,
                samples: 4 // Enable MSAA
            }
        );
    }

    init(bloomRenderTargets) {
        try {
            // Clean up existing resources if reinitializing
            if (this.initialized) {
                this.dispose();
            }

            if (!bloomRenderTargets) {
                throw new Error('No bloom render targets provided');
            }

            // Verify all required textures are available
            const requiredLayers = [LAYERS.BLOOM, LAYERS.HOLOGRAM, LAYERS.EDGE];
            requiredLayers.forEach(layer => {
                const target = bloomRenderTargets.get(layer);
                if (!target || !target.texture) {
                    throw new Error(`Missing bloom render target for layer ${layer}`);
                }
            });

            const renderTarget = this.createRenderTarget();
            this.composer = new EffectComposer(this.renderer, renderTarget);

            const shader = {
                uniforms: {
                    baseTexture: { value: null },
                    bloomTexture0: { value: bloomRenderTargets.get(LAYERS.BLOOM).texture },
                    bloomTexture1: { value: bloomRenderTargets.get(LAYERS.HOLOGRAM).texture },
                    bloomTexture2: { value: bloomRenderTargets.get(LAYERS.EDGE).texture }
                },
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D baseTexture;
                    uniform sampler2D bloomTexture0;
                    uniform sampler2D bloomTexture1;
                    uniform sampler2D bloomTexture2;
                    varying vec2 vUv;

                    void main() {
                        vec4 baseColor = texture2D(baseTexture, vUv);
                        vec3 bloomColor0 = texture2D(bloomTexture0, vUv).rgb;
                        vec3 bloomColor1 = texture2D(bloomTexture1, vUv).rgb;
                        vec3 bloomColor2 = texture2D(bloomTexture2, vUv).rgb;
                        
                        // Combine bloom layers with proper HDR handling
                        vec3 bloomSum = bloomColor0 + bloomColor1 + bloomColor2;
                        vec3 color = baseColor.rgb + bloomSum;
                        
                        // Apply tone mapping in shader for better control
                        color = color / (vec3(1.0) + color); // Simple Reinhard tone mapping
                        
                        // Gamma correction
                        color = pow(color, vec3(1.0 / 2.2));
                        
                        gl_FragColor = vec4(color, baseColor.a);
                    }
                `
            };

            const finalPass = new ShaderPass(new THREE.ShaderMaterial(shader));
            finalPass.renderToScreen = true;
            this.composer.addPass(finalPass);

            this.initialized = true;
        } catch (error) {
            console.error('Error initializing composition effect:', error);
            this.dispose();
        }
    }

    render(baseTexture) {
        if (!this.initialized || !this.composer) {
            return;
        }

        try {
            if (!baseTexture) {
                console.warn('No base texture provided for composition render');
                return;
            }

            const finalPass = this.composer.passes[0];
            if (finalPass && finalPass.uniforms) {
                finalPass.uniforms.baseTexture.value = baseTexture;
                this.composer.render();
            }
        } catch (error) {
            console.error('Error rendering composition effect:', error);
        }
    }

    resize(width, height) {
        if (!this.initialized || !this.composer) {
            return;
        }

        try {
            const pixelRatio = this.renderer.getPixelRatio();
            const actualWidth = Math.floor(width * pixelRatio);
            const actualHeight = Math.floor(height * pixelRatio);
            
            this.composer.setSize(actualWidth, actualHeight);
        } catch (error) {
            console.error('Error resizing composition effect:', error);
        }
    }

    dispose() {
        if (this.composer) {
            try {
                this.composer.dispose();
            } catch (error) {
                console.error('Error disposing composition effect:', error);
            }
        }
        this.composer = null;
        this.initialized = false;
    }
}
