import * as THREE from 'three';
import { BloomEffect } from './effects/BloomEffect.js';
import { CompositionEffect } from './effects/CompositionEffect.js';
import { LAYERS } from './layerManager.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';

export class EffectsManager {
    constructor(scene, camera, renderer) {
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
        
        // Effects
        this.bloomEffect = new BloomEffect(renderer, scene, camera);
        this.compositionEffect = new CompositionEffect(renderer);
        
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
        if (!this.renderer || !this.renderer.domElement) {
            console.warn('Renderer not ready, deferring post-processing initialization');
            return;
        }

        // Configure renderer
        this.renderer.autoClear = false;
        
        // Initialize effects with current settings
        const bloomSettings = visualizationSettings.getBloomSettings();
        
        // Initialize bloom first
        this.bloomEffect.init(bloomSettings);
        
        // Initialize composition effect with bloom render targets
        const bloomRenderTargets = this.bloomEffect.getRenderTargets();
        if (!bloomRenderTargets) {
            console.error('Failed to get bloom render targets');
            return;
        }
        
        // Initialize composition effect
        this.compositionEffect.init(bloomRenderTargets);
        
        console.log('Post-processing initialized successfully');
    }

    handleXRSessionStart() {
        const session = this.renderer.xr.getSession();
        if (session) {
            const baseLayer = session.renderState.baseLayer;
            const { width, height } = baseLayer.getViewport(session.views[0]);
            this.handleResize(width, height);
        }
    }

    handleXRSessionEnd() {
        this.handleResize(window.innerWidth, window.innerHeight);
    }
    
    render() {
        const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
        
        // Clear everything
        this.renderer.clear(true, true, true);

        // Render bloom layers
        this.bloomEffect.render(currentCamera);
        
        // Get base texture from bloom effect
        const bloomRenderTargets = this.bloomEffect.getRenderTargets();
        if (!bloomRenderTargets) {
            console.error('No bloom render targets available');
            return;
        }
        
        const baseTexture = bloomRenderTargets.get(LAYERS.BLOOM).texture;
        if (!baseTexture) {
            console.error('No base texture available');
            return;
        }
        
        // Reset camera to normal layer and render final composition
        currentCamera.layers.set(LAYERS.NORMAL_LAYER);
        this.compositionEffect.render(baseTexture);
    }
    
    handleResize(width = window.innerWidth, height = window.innerHeight) {
        this.bloomEffect.resize(width, height);
        this.compositionEffect.resize(width, height);
    }
    
    handleSettingsUpdate(event) {
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
    }
    
    dispose() {
        // Remove event listeners
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        this.renderer.xr.removeEventListener('sessionstart', this.handleXRSessionStart);
        this.renderer.xr.removeEventListener('sessionend', this.handleXRSessionEnd);
        
        // Restore original renderer settings
        this.renderer.autoClear = this.originalAutoClear;
        this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        
        // Dispose effects
        this.bloomEffect.dispose();
        this.compositionEffect.dispose();
    }
}
