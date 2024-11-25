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

        // Store original renderer state
        this.originalClearColor = this.renderer.getClearColor(new THREE.Color());
        this.originalClearAlpha = this.renderer.getClearAlpha();
    }

    createRenderTarget() {
        const pixelRatio = this.renderer.getPixelRatio();
        const width = Math.floor(window.innerWidth * pixelRatio);
        const height = Math.floor(window.innerHeight * pixelRatio);

        // Use HalfFloatType for HDR rendering
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

    init(settings) {
        if (this.initialized) {
            this.dispose();
        }

        if (!this.renderer || !this.renderer.domElement) {
            console.error('Renderer not ready for bloom effect initialization');
            return;
        }

        // Create base render target for scene
        const baseTarget = this.createRenderTarget();
        this.renderTargets.set('base', baseTarget);

        const layers = [
            {
                layer: LAYERS.BLOOM,
                settings: {
                    strength: settings.nodeBloomStrength * 2.0,
                    radius: settings.nodeBloomRadius * 0.5,
                    threshold: settings.nodeBloomThreshold * 0.5
                }
            },
            {
                layer: LAYERS.HOLOGRAM,
                settings: {
                    strength: settings.environmentBloomStrength * 2.5,
                    radius: settings.environmentBloomRadius * 0.8,
                    threshold: settings.environmentBloomThreshold * 0.4
                }
            },
            {
                layer: LAYERS.EDGE,
                settings: {
                    strength: settings.edgeBloomStrength * 1.5,
                    radius: settings.edgeBloomRadius * 0.7,
                    threshold: settings.edgeBloomThreshold * 0.6
                }
            }
        ];

        try {
            // Set renderer to linear color space for HDR
            this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

            // Create base composer for main scene
            const baseComposer = new EffectComposer(this.renderer, baseTarget);
            const baseRenderPass = new RenderPass(this.scene, this.camera);
            baseRenderPass.clear = true;
            baseComposer.addPass(baseRenderPass);
            this.composers.set('base', baseComposer);

            // Create bloom composers for each layer
            layers.forEach(({ layer, settings }) => {
                const renderTarget = this.createRenderTarget();
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
                
                bloomPass.highQualityBloom = true;
                bloomPass.gammaCorrectionInShader = true;
                
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
        if (!this.initialized || !currentCamera) return;

        try {
            // Store original camera layers
            const originalLayerMask = currentCamera.layers.mask;

            // Render base scene first
            currentCamera.layers.set(LAYERS.NORMAL_LAYER);
            this.composers.get('base').render();

            // Render bloom layers
            this.composers.forEach((composer, layer) => {
                if (layer !== 'base') {
                    currentCamera.layers.set(layer);
                    composer.render();
                }
            });

            // Restore camera layers
            currentCamera.layers.mask = originalLayerMask;
        } catch (error) {
            console.error('Error rendering bloom effect:', error);
        }
    }

    resize(width, height) {
        if (!this.initialized) return;

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
        
        if (this.renderer) {
            this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        }
        
        this.renderTargets.clear();
        this.composers.clear();
        this.initialized = false;
    }

    getRenderTargets() {
        if (!this.initialized) return null;
        return this.renderTargets;
    }
}
