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

        // Store original renderer state
        this.originalClearColor = this.renderer.getClearColor(new THREE.Color());
        this.originalClearAlpha = this.renderer.getClearAlpha();
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
                type: THREE.HalfFloatType,
                colorSpace: THREE.LinearSRGBColorSpace,
                stencilBuffer: false,
                depthBuffer: true,
                samples: this.renderer.capabilities.isWebGL2 ? 4 : 0
            }
        );
    }

    init(bloomRenderTargets) {
        try {
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

            const baseTexture = bloomRenderTargets.get('base')?.texture;
            if (!baseTexture) {
                throw new Error('Missing base render target');
            }

            const renderTarget = this.createRenderTarget();
            this.composer = new EffectComposer(this.renderer, renderTarget);

            const shader = {
                uniforms: {
                    baseTexture: { value: baseTexture },
                    bloomTexture0: { value: bloomRenderTargets.get(LAYERS.BLOOM).texture },
                    bloomTexture1: { value: bloomRenderTargets.get(LAYERS.HOLOGRAM).texture },
                    bloomTexture2: { value: bloomRenderTargets.get(LAYERS.EDGE).texture },
                    bloomStrength0: { value: 1.5 },
                    bloomStrength1: { value: 1.2 },
                    bloomStrength2: { value: 0.8 },
                    exposure: { value: 1.2 },
                    gamma: { value: 2.2 },
                    saturation: { value: 1.2 }
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
                    uniform float bloomStrength0;
                    uniform float bloomStrength1;
                    uniform float bloomStrength2;
                    uniform float exposure;
                    uniform float gamma;
                    uniform float saturation;
                    
                    varying vec2 vUv;

                    vec3 adjustSaturation(vec3 color, float saturation) {
                        float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
                        return mix(vec3(luminance), color, saturation);
                    }

                    vec3 toneMap(vec3 color) {
                        // ACES filmic tone mapping
                        float a = 2.51;
                        float b = 0.03;
                        float c = 2.43;
                        float d = 0.59;
                        float e = 0.14;
                        return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
                    }

                    void main() {
                        // Sample all textures in linear space
                        vec3 baseColor = texture2D(baseTexture, vUv).rgb;
                        vec3 bloom0 = texture2D(bloomTexture0, vUv).rgb * bloomStrength0;
                        vec3 bloom1 = texture2D(bloomTexture1, vUv).rgb * bloomStrength1;
                        vec3 bloom2 = texture2D(bloomTexture2, vUv).rgb * bloomStrength2;
                        
                        // Combine bloom layers
                        vec3 bloomSum = bloom0 + bloom1 + bloom2;
                        
                        // Add bloom to base color
                        vec3 hdrColor = baseColor + bloomSum;
                        
                        // Apply exposure
                        hdrColor *= exposure;
                        
                        // Tone mapping
                        vec3 color = toneMap(hdrColor);
                        
                        // Adjust saturation
                        color = adjustSaturation(color, saturation);
                        
                        // Gamma correction
                        color = pow(color, vec3(1.0 / gamma));
                        
                        gl_FragColor = vec4(color, 1.0);
                    }
                `
            };

            const finalPass = new ShaderPass(new THREE.ShaderMaterial(shader));
            finalPass.renderToScreen = true;
            finalPass.clear = false;
            this.composer.addPass(finalPass);

            this.initialized = true;
        } catch (error) {
            console.error('Error initializing composition effect:', error);
            this.dispose();
        }
    }

    render(baseTexture) {
        if (!this.initialized || !this.composer) return;

        try {
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
        if (!this.initialized || !this.composer) return;

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
        
        if (this.renderer) {
            this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        }
        
        this.composer = null;
        this.initialized = false;
    }
}
