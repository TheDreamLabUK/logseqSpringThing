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
        this.xrRenderTargets = new Map();
        this.initialized = false;
        this.isXRActive = false;

        // Store original renderer state
        this.originalClearColor = this.renderer.getClearColor(new THREE.Color());
        this.originalClearAlpha = this.renderer.getClearAlpha();

        // Bind XR session change handlers
        this.handleXRSessionStart = this.handleXRSessionStart.bind(this);
        this.handleXRSessionEnd = this.handleXRSessionEnd.bind(this);

        window.addEventListener('xrsessionstart', this.handleXRSessionStart);
        window.addEventListener('xrsessionend', this.handleXRSessionEnd);
    }

    handleXRSessionStart() {
        this.isXRActive = true;
        // Create XR-specific render targets if needed
        if (this.initialized) {
            this.createXRRenderTargets();
        }
    }

    handleXRSessionEnd() {
        this.isXRActive = false;
        // Clean up XR render targets
        this.xrRenderTargets.forEach(target => {
            if (target && target.dispose) {
                target.dispose();
            }
        });
        this.xrRenderTargets.clear();
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

    createXRRenderTargets() {
        const layers = [LAYERS.BLOOM, LAYERS.HOLOGRAM, LAYERS.EDGE];
        
        // Create base XR render target
        const baseTarget = this.createRenderTarget(true);
        this.xrRenderTargets.set('base', baseTarget);

        // Create XR render targets for each layer
        layers.forEach(layer => {
            const renderTarget = this.createRenderTarget(true);
            this.xrRenderTargets.set(layer, renderTarget);
        });
    }

    init(settings) {
        if (!settings) {
            console.error('No bloom settings provided');
            return;
        }

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

        // Create XR render targets if in XR mode
        if (this.isXRActive) {
            this.createXRRenderTargets();
        }

        // Adjust settings based on WebGL version and XR state
        const isWebGL2 = this.renderer.capabilities.isWebGL2;
        let adjustedSettings = { ...settings };

        if (!isWebGL2) {
            // Reduce quality for WebGL1
            adjustedSettings = {
                ...adjustedSettings,
                node_bloom_strength: settings.node_bloom_strength * 0.8,
                node_bloom_radius: settings.node_bloom_radius * 0.7,
                edge_bloom_strength: settings.edge_bloom_strength * 0.8,
                edge_bloom_radius: settings.edge_bloom_radius * 0.7,
                environment_bloom_strength: settings.environment_bloom_strength * 0.8,
                environment_bloom_radius: settings.environment_bloom_radius * 0.7
            };
        }

        if (this.isXRActive) {
            // Adjust bloom for XR
            adjustedSettings = {
                ...adjustedSettings,
                node_bloom_strength: adjustedSettings.node_bloom_strength * 1.2,
                node_bloom_radius: adjustedSettings.node_bloom_radius * 0.8,
                edge_bloom_strength: adjustedSettings.edge_bloom_strength * 1.2,
                edge_bloom_radius: adjustedSettings.edge_bloom_radius * 0.8
            };
        }

        const layers = [
            {
                layer: LAYERS.BLOOM,
                settings: {
                    strength: adjustedSettings.node_bloom_strength,
                    radius: adjustedSettings.node_bloom_radius,
                    threshold: adjustedSettings.node_bloom_threshold
                }
            },
            {
                layer: LAYERS.HOLOGRAM,
                settings: {
                    strength: adjustedSettings.environment_bloom_strength,
                    radius: adjustedSettings.environment_bloom_radius,
                    threshold: adjustedSettings.environment_bloom_threshold
                }
            },
            {
                layer: LAYERS.EDGE,
                settings: {
                    strength: adjustedSettings.edge_bloom_strength,
                    radius: adjustedSettings.edge_bloom_radius,
                    threshold: adjustedSettings.edge_bloom_threshold
                }
            }
        ];

        try {
            // Set renderer color space based on WebGL version
            this.renderer.outputColorSpace = isWebGL2 ? 
                THREE.LinearSRGBColorSpace : 
                THREE.SRGBColorSpace;

            // Create composers for both regular and XR rendering
            this.createComposers(layers, false); // Regular composers
            if (this.isXRActive) {
                this.createComposers(layers, true); // XR composers
            }

            this.initialized = true;
        } catch (error) {
            console.error('Error initializing bloom effect:', error);
            this.dispose();
        }
    }

    updateSettings(settings) {
        if (!this.initialized || !settings) return;

        try {
            const layers = [
                {
                    layer: LAYERS.BLOOM,
                    settings: {
                        strength: settings.node_bloom_strength,
                        radius: settings.node_bloom_radius,
                        threshold: settings.node_bloom_threshold
                    }
                },
                {
                    layer: LAYERS.HOLOGRAM,
                    settings: {
                        strength: settings.environment_bloom_strength,
                        radius: settings.environment_bloom_radius,
                        threshold: settings.environment_bloom_threshold
                    }
                },
                {
                    layer: LAYERS.EDGE,
                    settings: {
                        strength: settings.edge_bloom_strength,
                        radius: settings.edge_bloom_radius,
                        threshold: settings.edge_bloom_threshold
                    }
                }
            ];

            // Update bloom passes in composers
            layers.forEach(({ layer, settings }) => {
                const composer = this.composers.get(layer.toString());
                if (composer) {
                    const bloomPass = composer.passes.find(pass => pass instanceof UnrealBloomPass);
                    if (bloomPass) {
                        bloomPass.strength = settings.strength;
                        bloomPass.radius = settings.radius;
                        bloomPass.threshold = settings.threshold;
                    }
                }

                // Update XR composers if active
                if (this.isXRActive) {
                    const xrComposer = this.composers.get(`xr_${layer}`);
                    if (xrComposer) {
                        const bloomPass = xrComposer.passes.find(pass => pass instanceof UnrealBloomPass);
                        if (bloomPass) {
                            bloomPass.strength = settings.strength * 1.2; // Adjust for XR
                            bloomPass.radius = settings.radius * 0.8; // Adjust for XR
                            bloomPass.threshold = settings.threshold;
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error updating bloom settings:', error);
        }
    }

    createComposers(layers, isXR) {
        const targets = isXR ? this.xrRenderTargets : this.renderTargets;
        const composerPrefix = isXR ? 'xr_' : '';

        // Create base composer
        const baseComposer = new EffectComposer(this.renderer, targets.get('base'));
        const baseRenderPass = new RenderPass(this.scene, this.camera);
        baseRenderPass.clear = true;
        baseComposer.addPass(baseRenderPass);
        this.composers.set(`${composerPrefix}base`, baseComposer);

        // Create bloom composers for each layer
        layers.forEach(({ layer, settings }) => {
            const composer = new EffectComposer(this.renderer, targets.get(layer));
            composer.renderToScreen = false;
            
            const renderPass = new RenderPass(this.scene, this.camera);
            renderPass.clear = true;
            
            const bloomPass = new UnrealBloomPass(
                new THREE.Vector2(
                    targets.get(layer).width,
                    targets.get(layer).height
                ),
                settings.strength,
                settings.radius,
                settings.threshold
            );
            
            bloomPass.highQualityBloom = this.renderer.capabilities.isWebGL2;
            bloomPass.gammaCorrectionInShader = this.renderer.capabilities.isWebGL2;
            
            composer.addPass(renderPass);
            composer.addPass(bloomPass);
            
            this.composers.set(`${composerPrefix}${layer}`, composer);
        });
    }

    render(currentCamera) {
        if (!this.initialized || !currentCamera) return;

        try {
            const isXRFrame = this.renderer.xr.isPresenting;
            const composerPrefix = isXRFrame ? 'xr_' : '';
            
            // Store original camera layers
            const originalLayerMask = currentCamera.layers.mask;

            // Render base scene first
            currentCamera.layers.set(LAYERS.NORMAL_LAYER);
            this.composers.get(`${composerPrefix}base`).render();

            // Render bloom layers
            this.composers.forEach((composer, key) => {
                if (key.startsWith(composerPrefix) && !key.endsWith('base')) {
                    const layer = parseInt(key.split('_').pop());
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

        // Resize regular render targets and composers
        this.renderTargets.forEach(target => {
            if (target && target.setSize) {
                target.setSize(actualWidth, actualHeight);
            }
        });
        
        this.composers.forEach((composer, key) => {
            if (!key.startsWith('xr_') && composer && composer.setSize) {
                composer.setSize(actualWidth, actualHeight);
            }
        });

        // Don't resize XR targets - they're managed by the XR system
    }

    dispose() {
        // Remove event listeners
        window.removeEventListener('xrsessionstart', this.handleXRSessionStart);
        window.removeEventListener('xrsessionend', this.handleXRSessionEnd);

        // Dispose render targets
        this.renderTargets.forEach(target => {
            if (target && target.dispose) target.dispose();
        });
        
        this.xrRenderTargets.forEach(target => {
            if (target && target.dispose) target.dispose();
        });
        
        // Dispose composers
        this.composers.forEach(composer => {
            if (composer && composer.dispose) composer.dispose();
        });
        
        // Reset renderer state
        if (this.renderer) {
            this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        }
        
        // Clear collections
        this.renderTargets.clear();
        this.xrRenderTargets.clear();
        this.composers.clear();
        this.initialized = false;
        this.isXRActive = false;
    }

    getRenderTargets() {
        if (!this.initialized) return null;
        return this.isXRActive ? this.xrRenderTargets : this.renderTargets;
    }
}
