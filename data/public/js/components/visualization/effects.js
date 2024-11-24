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
        
        // Settings handler
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

        // XR session handlers
        this.renderer.xr.addEventListener('sessionstart', () => {
            this.isXRActive = true;
            this.handleXRSessionStart();
        });

        this.renderer.xr.addEventListener('sessionend', () => {
            this.isXRActive = false;
            this.handleXRSessionEnd();
        });
    }
    
    initPostProcessing() {
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
            
            // Initialize effects
            this.bloomEffect = new BloomEffect(this.renderer, this.scene, this.camera);
            this.compositionEffect = new CompositionEffect(this.renderer);
            
            // Initialize effects with current settings
            const bloomSettings = visualizationSettings.getBloomSettings();
            
            // Initialize bloom first
            this.bloomEffect.init(bloomSettings);
            
            // Initialize composition effect with bloom render targets
            const bloomRenderTargets = this.bloomEffect.getRenderTargets();
            if (!bloomRenderTargets) {
                throw new Error('Failed to get bloom render targets');
            }
            
            // Initialize composition effect
            this.compositionEffect.init(bloomRenderTargets);
            
            this.initialized = true;
            console.log('Post-processing initialized successfully');
        } catch (error) {
            console.error('Error initializing post-processing:', error);
            this.dispose();
        }
    }

    handleXRSessionStart() {
        try {
            const session = this.renderer.xr.getSession();
            if (session) {
                const baseLayer = session.renderState.baseLayer;
                const { width, height } = baseLayer.getViewport(session.views[0]);
                this.handleResize(width, height);
            }
        } catch (error) {
            console.error('Error handling XR session start:', error);
        }
    }

    handleXRSessionEnd() {
        this.handleResize(window.innerWidth, window.innerHeight);
    }
    
    render() {
        if (!this.initialized || !this.bloomEffect || !this.compositionEffect) {
            // If effects aren't initialized yet, do a normal render
            this.renderer.render(this.scene, this.camera);
            return;
        }

        try {
            const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
            
            // Clear everything
            this.renderer.clear(true, true, true);

            // Render bloom layers
            this.bloomEffect.render(currentCamera);
            
            // Get base texture from bloom effect
            const bloomRenderTargets = this.bloomEffect.getRenderTargets();
            if (!bloomRenderTargets) {
                throw new Error('No bloom render targets available');
            }
            
            const baseTexture = bloomRenderTargets.get(LAYERS.BLOOM).texture;
            if (!baseTexture) {
                throw new Error('No base texture available');
            }
            
            // Reset camera to normal layer and render final composition
            currentCamera.layers.set(LAYERS.NORMAL_LAYER);
            this.compositionEffect.render(baseTexture);
        } catch (error) {
            console.error('Error during effect rendering:', error);
            // Fallback to normal rendering
            this.renderer.render(this.scene, this.camera);
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
        }
    }
    
    handleSettingsUpdate(event) {
        if (!this.initialized) {
            return;
        }

        try {
            const settings = event.detail;
            if (settings.bloom) {
                const bloomSettings = {
                    nodeBloomStrength: settings.bloom.nodeStrength || 0.8,
                    nodeBloomRadius: settings.bloom.nodeRadius || 0.3,
                    nodeBloomThreshold: settings.bloom.nodeThreshold || 0.2,
                    edgeBloomStrength: settings.bloom.edgeStrength || 0.6,
                    edgeBloomRadius: settings.bloom.edgeRadius || 0.4,
                    edgeBloomThreshold: settings.bloom.edgeThreshold || 0.1,
                    environmentBloomStrength: settings.bloom.envStrength || 0.7,
                    environmentBloomRadius: settings.bloom.envRadius || 0.3,
                    environmentBloomThreshold: settings.bloom.envThreshold || 0.1
                };
                
                // Reinitialize bloom with new settings
                this.bloomEffect.init(bloomSettings);
                
                // Reinitialize composition effect with updated bloom render targets
                const bloomRenderTargets = this.bloomEffect.getRenderTargets();
                if (bloomRenderTargets) {
                    this.compositionEffect.init(bloomRenderTargets);
                }
            }
        } catch (error) {
            console.error('Error updating settings:', error);
        }
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
            
            this.initialized = false;
        } catch (error) {
            console.error('Error disposing effects:', error);
        }
    }
}
