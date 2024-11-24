import * as THREE from 'three';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { LAYERS } from '../layerManager.js';

export class BloomEffect {
    constructor(renderer, scene, camera) {
        if (!renderer || !renderer.domElement) {
            throw new Error('Invalid renderer provided to BloomEffect');
        }
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.composers = new Map();
        this.renderTargets = new Map();
        this.initialized = false;
    }

    createRenderTarget() {
        if (!this.renderer.capabilities.isWebGL2) {
            console.warn('WebGL 2 not available, some features may be limited');
        }

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

    init(settings) {
        // Clean up existing resources if reinitializing
        if (this.initialized) {
            this.dispose();
        }

        if (!this.renderer || !this.renderer.domElement) {
            console.error('Renderer not ready for bloom effect initialization');
            return;
        }

        const layers = [
            {
                layer: LAYERS.BLOOM,
                settings: {
                    strength: settings.nodeBloomStrength * 1.2,
                    radius: settings.nodeBloomRadius,
                    threshold: settings.nodeBloomThreshold * 0.8
                }
            },
            {
                layer: LAYERS.HOLOGRAM,
                settings: {
                    strength: settings.environmentBloomStrength * 1.5,
                    radius: settings.environmentBloomRadius * 1.2,
                    threshold: settings.environmentBloomThreshold * 0.7
                }
            },
            {
                layer: LAYERS.EDGE,
                settings: {
                    strength: settings.edgeBloomStrength * 1.3,
                    radius: settings.edgeBloomRadius,
                    threshold: settings.edgeBloomThreshold * 0.9
                }
            }
        ];

        try {
            layers.forEach(({ layer, settings }) => {
                const renderTarget = this.createRenderTarget();
                if (!renderTarget) {
                    throw new Error('Failed to create render target');
                }
                this.renderTargets.set(layer, renderTarget);
                
                const composer = new EffectComposer(this.renderer, renderTarget);
                composer.renderToScreen = false;
                
                const renderPass = new RenderPass(this.scene, this.camera);
                renderPass.clear = true;
                
                const bloomPass = new UnrealBloomPass(
                    new THREE.Vector2(window.innerWidth, window.innerHeight),
                    settings.strength,
                    settings.radius,
                    settings.threshold
                );
                
                composer.addPass(renderPass);
                composer.addPass(bloomPass);
                
                this.composers.set(layer, composer);
            });

            this.initialized = true;
        } catch (error) {
            console.error('Error initializing bloom effect:', error);
            this.dispose();
        }
    }

    render(currentCamera) {
        if (!this.initialized || !currentCamera) {
            return;
        }

        try {
            this.composers.forEach((composer, layer) => {
                const originalLayerMask = currentCamera.layers.mask;
                currentCamera.layers.set(layer);
                
                if (composer.outputBuffer) {
                    composer.render();
                }
                
                currentCamera.layers.mask = originalLayerMask;
            });
        } catch (error) {
            console.error('Error rendering bloom effect:', error);
        }
    }

    resize(width, height) {
        if (!this.initialized) {
            return;
        }

        const pixelRatio = this.renderer.getPixelRatio();
        const actualWidth = Math.floor(width * pixelRatio);
        const actualHeight = Math.floor(height * pixelRatio);

        this.renderTargets.forEach(target => {
            if (target && target.setSize) {
                target.setSize(actualWidth, actualHeight);
            }
        });
        
        this.composers.forEach(composer => {
            if (composer && composer.setSize) {
                composer.setSize(actualWidth, actualHeight);
            }
        });
    }

    dispose() {
        this.renderTargets.forEach(target => {
            if (target && target.dispose) {
                target.dispose();
            }
        });
        
        this.composers.forEach(composer => {
            if (composer && composer.dispose) {
                composer.dispose();
            }
        });
        
        this.renderTargets.clear();
        this.composers.clear();
        this.initialized = false;
    }

    getRenderTargets() {
        if (!this.initialized) {
            return null;
        }
        return this.renderTargets;
    }
}
