import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { LAYERS } from './layerManager.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';

export class EffectsManager {
    constructor(scene, camera, renderer, settings = {}) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        // Store original renderer settings
        this.originalClearColor = this.renderer.getClearColor(new THREE.Color());
        this.originalClearAlpha = this.renderer.getClearAlpha();
        this.originalAutoClear = this.renderer.autoClear;
        
        // Composers for each layer
        this.composers = new Map();
        this.finalComposer = null;
        this.baseComposer = null;
        
        // Create hologram group
        this.hologramGroup = new THREE.Group();
        this.scene.add(this.hologramGroup);
        
        // Get settings
        this.bloomSettings = visualizationSettings.getBloomSettings();
        this.hologramSettings = visualizationSettings.getHologramSettings();
        
        // Bind settings update handler
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

        // XR-specific properties
        this.xrRenderTarget = null;
        this.isXRActive = false;
    }
    
    initPostProcessing() {
        if (!this.renderer || !this.renderer.domElement) {
            console.warn('Renderer not ready, deferring post-processing initialization');
            return;
        }

        // Configure renderer for post-processing
        this.renderer.autoClear = false;
        
        // Create render targets with HDR format
        const renderTarget = new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.HalfFloatType,
                encoding: THREE.sRGBEncoding,
                stencilBuffer: false,
                depthBuffer: true
            }
        );

        // Create XR-compatible render target
        this.xrRenderTarget = renderTarget.clone();
        
        // Create base composer for scene
        this.baseComposer = new EffectComposer(this.renderer, renderTarget.clone());
        const basePass = new RenderPass(this.scene, this.camera);
        basePass.clear = true;
        this.baseComposer.addPass(basePass);
        
        // Create bloom composers for each layer
        const layers = [
            {
                layer: LAYERS.BLOOM,
                settings: {
                    strength: this.bloomSettings.nodeBloomStrength * 1.2,
                    radius: this.bloomSettings.nodeBloomRadius,
                    threshold: this.bloomSettings.nodeBloomThreshold * 0.8
                }
            },
            {
                layer: LAYERS.HOLOGRAM,
                settings: {
                    strength: this.bloomSettings.environmentBloomStrength * 1.5,
                    radius: this.bloomSettings.environmentBloomRadius * 1.2,
                    threshold: this.bloomSettings.environmentBloomThreshold * 0.7
                }
            },
            {
                layer: LAYERS.EDGE,
                settings: {
                    strength: this.bloomSettings.edgeBloomStrength * 1.3,
                    radius: this.bloomSettings.edgeBloomRadius,
                    threshold: this.bloomSettings.edgeBloomThreshold * 0.9
                }
            }
        ];
        
        // Create composers for each layer
        layers.forEach(({ layer, settings }) => {
            const composer = new EffectComposer(this.renderer, renderTarget.clone());
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

        // Create final composer
        this.finalComposer = new EffectComposer(this.renderer, renderTarget.clone());
        
        // Add custom shader pass to combine base scene and bloom layers
        const finalPass = new ShaderPass(
            new THREE.ShaderMaterial({
                uniforms: {
                    baseTexture: { value: this.baseComposer.renderTarget2.texture },
                    bloomTexture0: { value: this.composers.get(LAYERS.BLOOM).renderTarget2.texture },
                    bloomTexture1: { value: this.composers.get(LAYERS.HOLOGRAM).renderTarget2.texture },
                    bloomTexture2: { value: this.composers.get(LAYERS.EDGE).renderTarget2.texture }
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
                    varying vec2 vUv;

                    void main() {
                        vec4 base = texture2D(baseTexture, vUv);
                        vec4 bloom0 = texture2D(bloomTexture0, vUv);
                        vec4 bloom1 = texture2D(bloomTexture1, vUv);
                        vec4 bloom2 = texture2D(bloomTexture2, vUv);
                        
                        // Start with base scene color
                        vec3 color = base.rgb;
                        
                        // Add bloom layers
                        color += bloom0.rgb;
                        color += bloom1.rgb;
                        color += bloom2.rgb;
                        
                        // HDR tone mapping
                        color = color / (vec3(1.0) + color);
                        
                        // Gamma correction
                        color = pow(color, vec3(1.0 / 2.2));
                        
                        gl_FragColor = vec4(color, base.a);
                    }
                `,
                transparent: true,
                depthWrite: false,
                depthTest: true
            })
        );
        finalPass.clear = false;
        this.finalComposer.addPass(finalPass);

        // Create hologram structure after composers are ready
        this.createHologramStructure();

        // Set up XR session listeners
        this.renderer.xr.addEventListener('sessionstart', () => {
            this.isXRActive = true;
            this.handleXRSessionStart();
        });

        this.renderer.xr.addEventListener('sessionend', () => {
            this.isXRActive = false;
            this.handleXRSessionEnd();
        });
    }

    handleXRSessionStart() {
        // Update render targets for XR
        const session = this.renderer.xr.getSession();
        if (session) {
            const baseLayer = session.renderState.baseLayer;
            const { width, height } = baseLayer.getViewport(session.views[0]);
            
            this.composers.forEach(composer => {
                composer.setSize(width, height);
            });
            this.baseComposer.setSize(width, height);
            this.finalComposer.setSize(width, height);
        }
    }

    handleXRSessionEnd() {
        // Reset to normal rendering
        this.handleResize();
    }

    createHologramStructure() {
        // ... (hologram structure code remains unchanged)
    }

    animate() {
        // ... (animation code remains unchanged)
    }
    
    render() {
        const currentCamera = this.isXRActive ? this.renderer.xr.getCamera() : this.camera;
        
        // Clear everything
        this.renderer.clear(true, true, true);

        // Render base scene
        currentCamera.layers.set(LAYERS.NORMAL_LAYER);
        this.baseComposer.render();

        // Render bloom layers
        this.composers.forEach((composer, layer) => {
            currentCamera.layers.set(layer);
            composer.render();
        });
        
        // Reset camera layers and render final composition
        currentCamera.layers.set(LAYERS.NORMAL_LAYER);
        this.finalComposer.render();
    }
    
    handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Resize all composers
        this.composers.forEach(composer => {
            composer.setSize(width, height);
        });
        
        if (this.baseComposer) {
            this.baseComposer.setSize(width, height);
        }
        
        if (this.finalComposer) {
            this.finalComposer.setSize(width, height);
        }
    }
    
    updateBloom(settings) {
        // ... (updateBloom code remains unchanged)
    }
    
    handleSettingsUpdate(event) {
        // ... (handleSettingsUpdate code remains unchanged)
    }
    
    dispose() {
        // Restore original renderer settings
        this.renderer.autoClear = this.originalAutoClear;
        this.renderer.setClearColor(this.originalClearColor, this.originalClearAlpha);
        
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        
        // Dispose all composers
        this.composers.forEach(composer => {
            composer.dispose();
        });
        
        if (this.baseComposer) {
            this.baseComposer.dispose();
        }
        
        if (this.finalComposer) {
            this.finalComposer.dispose();
        }

        // Dispose hologram resources
        this.hologramGroup.children.forEach(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
        this.scene.remove(this.hologramGroup);

        // Dispose XR render target
        if (this.xrRenderTarget) {
            this.xrRenderTarget.dispose();
        }
    }
}
