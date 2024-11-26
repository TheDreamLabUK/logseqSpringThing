import * as THREE from 'three';
import { BloomEffect } from './effects/BloomEffect.js';
import { CompositionEffect } from './effects/CompositionEffect.js';
import { LAYERS } from './layerManager.js';
import { visualizationSettings } from '../services/visualizationSettings.js';

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
        
        // Initialize effects
        this.bloomEffect = null;
        this.compositionEffect = null;
        this.initialized = false;
        this.pendingInitialization = true;
        
        // XR properties
        this.isXRActive = false;
        
        // Bind methods
        this.render = this.render.bind(this);
        this.handleResize = this.handleResize.bind(this);
        this.handleXRSessionStart = this.handleXRSessionStart.bind(this);
        this.handleXRSessionEnd = this.handleXRSessionEnd.bind(this);
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        this.dispose = this.dispose.bind(this);
        
        // Add event listeners
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        window.addEventListener('xrsessionstart', this.handleXRSessionStart);
        window.addEventListener('xrsessionend', this.handleXRSessionEnd);
    }
    
    async initPostProcessing(isXR = false) {
        try {
            console.log('Starting post-processing initialization');
            if (!this.renderer || !this.renderer.domElement) {
                throw new Error('Renderer not ready for post-processing initialization');
            }

            // Wait for settings if needed
            if (this.pendingInitialization) {
                console.log('Waiting for settings before initializing effects...');
                return false;
            }

            // Clean up existing effects if reinitializing
            if (this.initialized) {
                this.dispose();
            }

            // Configure renderer
            this.renderer.autoClear = true;
            this.isXRActive = isXR;

            // Get settings from server
            const settings = visualizationSettings.getSettings();
            if (!settings?.bloom) {
                console.error('Bloom settings not available from server');
                return false;
            }

            // Initialize bloom effect with server settings
            this.bloomEffect = new BloomEffect(this.renderer, this.scene, this.camera);
            await this.bloomEffect.init(settings.bloom);

            // Initialize composition effect
            this.compositionEffect = new CompositionEffect(this.renderer);
            await this.compositionEffect.init(this.bloomEffect.getRenderTargets());
            
            this.initialized = true;
            console.log('Post-processing initialization complete with server settings');
            return true;
        } catch (error) {
            console.error('Error in post-processing initialization:', error);
            this.dispose();
            return false;
        }
    }

    handleXRSessionStart() {
        console.log('XR session started');
        this.isXRActive = true;
        if (this.initialized) {
            // Reinitialize effects for XR
            this.initPostProcessing(true);
        }
    }

    handleXRSessionEnd() {
        console.log('XR session ended');
        this.isXRActive = false;
        if (this.initialized) {
            // Reinitialize effects for non-XR
            this.initPostProcessing(false);
        }
    }
    
    render() {
        if (!this.initialized) {
            // Fallback to direct rendering if effects aren't initialized
            this.renderer.render(this.scene, this.camera);
            return;
        }

        try {
            const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
            
            // Render bloom passes
            if (this.bloomEffect) {
                this.bloomEffect.render(currentCamera);
            }

            // Render final composition
            if (this.compositionEffect) {
                const baseTexture = this.bloomEffect.getRenderTargets().get('base').texture;
                this.compositionEffect.render(baseTexture);
            }
        } catch (error) {
            console.error('Error during rendering:', error);
            // Fallback to direct rendering on error
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    handleResize(width = window.innerWidth, height = window.innerHeight) {
        if (!this.initialized) return;

        try {
            // Update renderer size
            this.renderer.setSize(width, height);

            // Update effect sizes
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
        console.log('Received settings update:', event.detail);

        // If waiting for initial settings, initialize
        if (this.pendingInitialization) {
            this.pendingInitialization = false;
            this.initPostProcessing(this.isXRActive);
            return;
        }

        // Update existing effects with new settings
        if (!this.initialized) return;

        try {
            const settings = event.detail;
            
            if (settings.bloom && this.bloomEffect) {
                this.bloomEffect.updateSettings(settings.bloom);
                if (this.compositionEffect) {
                    this.compositionEffect.updateSettings(this.bloomEffect.getRenderTargets());
                }
            }

            if (settings.visualization?.material && this.bloomEffect) {
                this.bloomEffect.updateMaterialSettings(settings.visualization.material);
            }
        } catch (error) {
            console.error('Error handling settings update:', error);
        }
    }
    
    dispose() {
        // Remove event listeners
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        window.removeEventListener('xrsessionstart', this.handleXRSessionStart);
        window.removeEventListener('xrsessionend', this.handleXRSessionEnd);
        
        // Dispose effects
        if (this.bloomEffect) {
            this.bloomEffect.dispose();
            this.bloomEffect = null;
        }
        
        if (this.compositionEffect) {
            this.compositionEffect.dispose();
            this.compositionEffect = null;
        }
        
        // Restore renderer settings
        if (this.renderer) {
            this.renderer.autoClear = this.originalAutoClear;
            this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        }
        
        this.initialized = false;
        this.isXRActive = false;
    }

    isInitialized() {
        return this.initialized;
    }

    isXREnabled() {
        return this.isXRActive;
    }
}
