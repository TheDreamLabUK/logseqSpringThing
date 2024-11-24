import * as THREE from 'three';
import { BloomEffect } from './effects/BloomEffect.js';
import { CompositionEffect } from './effects/CompositionEffect.js';
import { LAYERS } from './layerManager.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';

export class EffectsManager {
    constructor(scene, camera, renderer) {
        if (!renderer || !renderer.domElement) {
            throw new Error('Invalid renderer provided to EffectsManager');
        }
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        // Store original renderer settings
        this.originalClearColor = this.renderer.getClearColor(new THREE.Color());
        this.originalClearAlpha = this.renderer.getClearAlpha();
        this.originalAutoClear = this.renderer.autoClear;
        
        // Configure renderer for optimal performance
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        // Initialize effects as null
        this.bloomEffect = null;
        this.compositionEffect = null;
        this.initialized = false;
        
        // XR properties
        this.isXRActive = false;
        this.xrRenderTargets = new Map();
        
        // Bind methods that need binding
        this.render = this.render.bind(this);
        this.handleResize = this.handleResize.bind(this);
        this.updateBloom = this.updateBloom.bind(this);
        this.updateFisheye = this.updateFisheye.bind(this);
        this.dispose = this.dispose.bind(this);
        
        // Settings handler - bind and add listener
        this.handleSettingsUpdate = (event) => {
            if (!this.initialized) return;
            try {
                const settings = event.detail;
                if (settings.bloom) {
                    this.updateBloom(settings.bloom);
                }
                if (settings.fisheye) {
                    this.updateFisheye(settings.fisheye);
                }
            } catch (error) {
                console.error('Error handling settings update:', error);
            }
        };
        
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
    }
    
    async initPostProcessing(isXR = false) {
        try {
            if (!this.renderer || !this.renderer.domElement) {
                throw new Error('Renderer not ready for post-processing initialization');
            }

            // Clean up existing effects if reinitializing
            if (this.initialized) {
                this.dispose();
            }

            // Configure renderer
            this.renderer.autoClear = false;
            this.isXRActive = isXR;
            
            // Initialize effects with mode-specific settings
            const bloomSettings = visualizationSettings.getBloomSettings();
            if (isXR) {
                // Adjust settings for XR
                bloomSettings.nodeBloomStrength *= 0.8;
                bloomSettings.edgeBloomRadius *= 0.7;
                bloomSettings.environmentBloomThreshold *= 1.2;
            }
            
            // Initialize bloom first
            this.bloomEffect = new BloomEffect(this.renderer, this.scene, this.camera);
            await this.bloomEffect.init(bloomSettings);
            
            // Get bloom render targets
            const bloomRenderTargets = this.bloomEffect.getRenderTargets();
            if (!bloomRenderTargets) {
                throw new Error('Failed to get bloom render targets');
            }
            
            // Store XR-specific render targets if needed
            if (isXR) {
                this.xrRenderTargets = bloomRenderTargets;
            }
            
            // Initialize composition effect
            this.compositionEffect = new CompositionEffect(this.renderer);
            await this.compositionEffect.init(bloomRenderTargets);
            
            this.initialized = true;
            console.log(`Post-processing initialized successfully for ${isXR ? 'XR' : 'desktop'} mode`);
            return true;
        } catch (error) {
            console.error('Error initializing post-processing:', error);
            this.dispose();
            return false;
        }
    }
    
    render() {
        if (!this.initialized || !this.bloomEffect || !this.compositionEffect) {
            throw new Error('Effects not properly initialized');
        }

        try {
            const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
            
            // Clear everything
            this.renderer.clear(true, true, true);

            // Render bloom layers
            this.bloomEffect.render(currentCamera);
            
            // Get appropriate render targets
            const renderTargets = this.isXRActive ? this.xrRenderTargets : this.bloomEffect.getRenderTargets();
            if (!renderTargets) {
                throw new Error('No render targets available');
            }
            
            const baseTexture = renderTargets.get(LAYERS.BLOOM).texture;
            if (!baseTexture) {
                throw new Error('No base texture available');
            }
            
            // Reset camera to normal layer and render final composition
            currentCamera.layers.set(LAYERS.NORMAL_LAYER);
            this.compositionEffect.render(baseTexture);
        } catch (error) {
            throw new Error(`Error during effect rendering: ${error.message}`);
        }
    }
    
    handleResize(width = window.innerWidth, height = window.innerHeight) {
        if (!this.initialized) {
            return;
        }

        try {
            if (this.bloomEffect) {
                this.bloomEffect.resize(width, height);
            }
            if (this.compositionEffect) {
                this.compositionEffect.resize(width, height);
            }
        } catch (error) {
            console.error('Error handling resize:', error);
            throw error;
        }
    }
    
    updateBloom(settings) {
        if (!this.initialized || !this.bloomEffect) {
            return;
        }

        try {
            // Adjust settings for XR if needed
            if (this.isXRActive) {
                settings = {
                    ...settings,
                    nodeBloomStrength: settings.nodeBloomStrength * 0.8,
                    edgeBloomRadius: settings.edgeBloomRadius * 0.7,
                    environmentBloomThreshold: settings.environmentBloomThreshold * 1.2
                };
            }
            
            // Reinitialize bloom with new settings
            this.bloomEffect.init(settings);
            
            // Reinitialize composition effect with updated bloom render targets
            const bloomRenderTargets = this.bloomEffect.getRenderTargets();
            if (bloomRenderTargets) {
                if (this.isXRActive) {
                    this.xrRenderTargets = bloomRenderTargets;
                }
                this.compositionEffect.init(bloomRenderTargets);
            }
        } catch (error) {
            console.error('Error updating bloom settings:', error);
            throw error;
        }
    }
    
    updateFisheye(settings) {
        // Placeholder for future fisheye effect implementation
        console.log('Fisheye effect not yet implemented');
    }
    
    dispose() {
        try {
            // Remove event listeners
            window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
            
            // Restore original renderer settings
            if (this.renderer) {
                this.renderer.autoClear = this.originalAutoClear;
                this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
            }
            
            // Dispose effects
            if (this.bloomEffect) {
                this.bloomEffect.dispose();
                this.bloomEffect = null;
            }
            if (this.compositionEffect) {
                this.compositionEffect.dispose();
                this.compositionEffect = null;
            }
            
            // Clear XR render targets
            this.xrRenderTargets.clear();
            
            this.initialized = false;
        } catch (error) {
            console.error('Error disposing effects:', error);
        }
    }
}
