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
            console.log('Starting post-processing initialization');
            if (!this.renderer || !this.renderer.domElement) {
                throw new Error('Renderer not ready for post-processing initialization');
            }

            // Clean up existing effects if reinitializing
            if (this.initialized) {
                this.dispose();
            }

            // Configure renderer
            this.renderer.autoClear = true; // Changed to true for direct rendering
            this.isXRActive = isXR;
            
            // Initialize with basic settings
            this.initialized = true;
            console.log('Basic rendering initialization complete');
            return true;
        } catch (error) {
            console.error('Error in post-processing initialization:', error);
            this.dispose();
            return false;
        }
    }
    
    render() {
        try {
            const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
            
            // Simple direct rendering without effects
            this.renderer.render(this.scene, currentCamera);
        } catch (error) {
            console.error('Error during rendering:', error);
            throw error;
        }
    }
    
    handleResize(width = window.innerWidth, height = window.innerHeight) {
        if (!this.initialized) {
            return;
        }

        try {
            // Update renderer size
            this.renderer.setSize(width, height);
        } catch (error) {
            console.error('Error handling resize:', error);
            throw error;
        }
    }
    
    updateBloom(settings) {
        // Disabled for now
        console.log('Bloom effects temporarily disabled');
    }
    
    updateFisheye(settings) {
        // Disabled for now
        console.log('Fisheye effect temporarily disabled');
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
            
            // Clear XR render targets
            this.xrRenderTargets.clear();
            
            this.initialized = false;
        } catch (error) {
            console.error('Error disposing effects:', error);
        }
    }
}
