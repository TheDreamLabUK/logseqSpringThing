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
        this.xrComposer = null;
        this.initialized = false;
        this.isXRActive = false;

        // Store original renderer state
        this.originalClearColor = this.renderer.getClearColor(new THREE.Color());
        this.originalClearAlpha = this.renderer.getClearAlpha();

        // Bind XR session handlers
        this.handleXRSessionStart = this.handleXRSessionStart.bind(this);
        this.handleXRSessionEnd = this.handleXRSessionEnd.bind(this);

        window.addEventListener('xrsessionstart', this.handleXRSessionStart);
        window.addEventListener('xrsessionend', this.handleXRSessionEnd);
    }

    handleXRSessionStart() {
        this.isXRActive = true;
        if (this.initialized) {
            // Create XR-specific composer if needed
            this.createXRComposer();
        }
    }

    handleXRSessionEnd() {
        this.isXRActive = false;
        if (this.xrComposer) {
            this.xrComposer.dispose();
            this.xrComposer = null;
        }
    }

    createRenderTarget(isXR = false) {
        let width, height;
        
        if (isXR && this.renderer.xr.getSession()) {
            const glProperties = this.renderer.properties.get(this.renderer.xr.getSession());
            const renderWidth = glProperties?.renderWidth || window.innerWidth * 2;
            const renderHeight = glProperties?.renderHeight || window.innerHeight;
            width = renderWidth;
            height = renderHeight;
        } else {
            const pixelRatio = this.renderer.getPixelRatio();
            width = Math.floor(window.innerWidth * pixelRatio);
            height = Math.floor(window.innerHeight * pixelRatio);
        }

        const isWebGL2 = this.renderer.capabilities.isWebGL2;
        return new THREE.WebGLRenderTarget(
            width,
            height,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: isWebGL2 ? THREE.HalfFloatType : THREE.UnsignedByteType,
                colorSpace: isWebGL2 ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace,
                stencilBuffer: false,
                depthBuffer: true,
                samples: isWebGL2 ? 4 : 0
            }
        );
    }

    createCompositionShader(isXR = false) {
        return {
            uniforms: {
                baseTexture: { value: null },
                bloomTexture0: { value: null },
                bloomTexture1: { value: null },
                bloomTexture2: { value: null },
                bloomStrength0: { value: 1.5 },
                bloomStrength1: { value: 1.2 },
                bloomStrength2: { value: 0.8 },
                exposure: { value: isXR ? 1.0 : 1.2 }, // Reduced exposure for XR
                gamma: { value: 2.2 },
                saturation: { value: isXR ? 1.1 : 1.2 }, // Reduced saturation for XR
                isXR: { value: isXR ? 1.0 : 0.0 }
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
                uniform float isXR;
                
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
                    // Adjust UV for XR if needed
                    vec2 adjustedUV = vUv;
                    if (isXR > 0.5) {
                        // Handle stereo rendering
                        adjustedUV.x = adjustedUV.x * 0.5;
                        if (gl_FragCoord.x > gl_FragCoord.w) {
                            adjustedUV.x += 0.5;
                        }
                    }
                    
                    // Sample all textures in linear space
                    vec3 baseColor = texture2D(baseTexture, adjustedUV).rgb;
                    vec3 bloom0 = texture2D(bloomTexture0, adjustedUV).rgb * bloomStrength0;
                    vec3 bloom1 = texture2D(bloomTexture1, adjustedUV).rgb * bloomStrength1;
                    vec3 bloom2 = texture2D(bloomTexture2, adjustedUV).rgb * bloomStrength2;
                    
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
    }

    createComposer(bloomRenderTargets, isXR = false) {
        const renderTarget = this.createRenderTarget(isXR);
        const composer = new EffectComposer(this.renderer, renderTarget);

        const shader = this.createCompositionShader(isXR);
        shader.uniforms.baseTexture.value = bloomRenderTargets.get('base').texture;
        shader.uniforms.bloomTexture0.value = bloomRenderTargets.get(LAYERS.BLOOM).texture;
        shader.uniforms.bloomTexture1.value = bloomRenderTargets.get(LAYERS.HOLOGRAM).texture;
        shader.uniforms.bloomTexture2.value = bloomRenderTargets.get(LAYERS.EDGE).texture;

        const finalPass = new ShaderPass(new THREE.ShaderMaterial(shader));
        finalPass.renderToScreen = true;
        finalPass.clear = false;
        composer.addPass(finalPass);

        return composer;
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

            // Create regular composer
            this.composer = this.createComposer(bloomRenderTargets, false);

            // Create XR composer if in XR mode
            if (this.isXRActive) {
                this.createXRComposer(bloomRenderTargets);
            }

            this.initialized = true;
        } catch (error) {
            console.error('Error initializing composition effect:', error);
            this.dispose();
        }
    }

    createXRComposer(bloomRenderTargets) {
        if (!this.renderer.xr.getSession()) return;
        this.xrComposer = this.createComposer(bloomRenderTargets, true);
    }

    render(baseTexture) {
        if (!this.initialized) return;

        try {
            const activeComposer = this.isXRActive ? this.xrComposer : this.composer;
            if (!activeComposer) return;

            const finalPass = activeComposer.passes[0];
            if (finalPass && finalPass.uniforms) {
                finalPass.uniforms.baseTexture.value = baseTexture;
                activeComposer.render();
            }
        } catch (error) {
            console.error('Error rendering composition effect:', error);
        }
    }

    resize(width, height) {
        if (!this.initialized) return;

        try {
            const pixelRatio = this.renderer.getPixelRatio();
            const actualWidth = Math.floor(width * pixelRatio);
            const actualHeight = Math.floor(height * pixelRatio);
            
            // Only resize non-XR composer
            if (this.composer) {
                this.composer.setSize(actualWidth, actualHeight);
            }
        } catch (error) {
            console.error('Error resizing composition effect:', error);
        }
    }

    dispose() {
        // Remove event listeners
        window.removeEventListener('xrsessionstart', this.handleXRSessionStart);
        window.removeEventListener('xrsessionend', this.handleXRSessionEnd);

        // Dispose composers
        if (this.composer) {
            this.composer.dispose();
            this.composer = null;
        }
        
        if (this.xrComposer) {
            this.xrComposer.dispose();
            this.xrComposer = null;
        }
        
        if (this.renderer) {
            this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        }
        
        this.initialized = false;
        this.isXRActive = false;
    }
}
