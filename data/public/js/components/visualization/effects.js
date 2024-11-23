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
        
        // Composers for each layer
        this.composers = new Map();
        this.finalComposer = null;
        
        // Create hologram group
        this.hologramGroup = new THREE.Group();
        this.scene.add(this.hologramGroup);
        
        // Get settings
        this.bloomSettings = visualizationSettings.getBloomSettings();
        this.hologramSettings = visualizationSettings.getHologramSettings();
        
        // Bind settings update handler
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
    }
    
    initPostProcessing() {
        if (!this.renderer || !this.renderer.domElement) {
            console.warn('Renderer not ready, deferring post-processing initialization');
            return;
        }

        // Create render targets with HDR format
        const renderTarget = new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.HalfFloatType,
                encoding: THREE.sRGBEncoding
            }
        );
        
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
        this.finalComposer = new EffectComposer(this.renderer);
        
        // Add render pass for base scene
        const renderPass = new RenderPass(this.scene, this.camera);
        this.finalComposer.addPass(renderPass);

        // Add custom shader pass to combine bloom layers
        const finalPass = new ShaderPass(
            new THREE.ShaderMaterial({
                uniforms: {
                    baseTexture: { value: null },
                    bloomTexture0: { value: this.composers.get(LAYERS.BLOOM).renderTarget2.texture },
                    bloomTexture1: { value: this.composers.get(LAYERS.HOLOGRAM).renderTarget2.texture },
                    bloomTexture2: { value: this.composers.get(LAYERS.EDGE).renderTarget2.texture },
                    bloomStrength0: { value: 1.0 },
                    bloomStrength1: { value: 0.8 },
                    bloomStrength2: { value: 0.6 }
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
                    varying vec2 vUv;

                    void main() {
                        vec4 base = texture2D(baseTexture, vUv);
                        vec4 bloom = vec4(0.0);
                        
                        // Combine bloom layers with weights
                        bloom += texture2D(bloomTexture0, vUv) * bloomStrength0;
                        bloom += texture2D(bloomTexture1, vUv) * bloomStrength1;
                        bloom += texture2D(bloomTexture2, vUv) * bloomStrength2;
                        
                        // HDR tone mapping
                        vec3 color = base.rgb + bloom.rgb;
                        color = color / (vec3(1.0) + color);
                        
                        // Gamma correction
                        color = pow(color, vec3(1.0 / 2.2));
                        
                        // Enhance contrast slightly
                        color = mix(vec3(0.0), color, 1.1);
                        
                        gl_FragColor = vec4(color, base.a);
                    }
                `
            }),
            "baseTexture"
        );
        
        this.finalComposer.addPass(finalPass);

        // Create hologram structure after composers are ready
        this.createHologramStructure();
    }

    createHologramStructure() {
        // Clear existing hologram structure
        while (this.hologramGroup.children.length > 0) {
            const child = this.hologramGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.hologramGroup.remove(child);
        }

        const hologramColor = new THREE.Color(this.hologramSettings.color);
        const hologramScale = this.hologramSettings.scale;
        const hologramOpacity = this.hologramSettings.opacity;

        // Create multiple rings with different sizes
        const ringSizes = [40, 30, 20];
        for (let i = 0; i < 3; i++) {
            const ringGeometry = new THREE.TorusGeometry(ringSizes[i], 3, 32, 100);
            const ringMaterial = new THREE.MeshPhysicalMaterial({
                color: hologramColor,
                emissive: hologramColor,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: hologramOpacity,
                metalness: 0.7,
                roughness: 0.2,
                clearcoat: 1.0,
                clearcoatRoughness: 0.1,
                toneMapped: false
            });

            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2 * i;
            ring.rotation.y = Math.PI / 4 * i;
            ring.userData.rotationSpeed = 0.002 * (i + 1);
            ring.layers.set(LAYERS.HOLOGRAM);
            this.hologramGroup.add(ring);
        }

        // Scale the entire hologram group
        this.hologramGroup.scale.setScalar(hologramScale);
    }

    animate() {
        // Animate hologram elements
        this.hologramGroup.children.forEach(child => {
            if (child.userData.rotationSpeed) {
                child.rotation.x += child.userData.rotationSpeed;
                child.rotation.y += child.userData.rotationSpeed;
            }
        });
    }
    
    render() {
        // Render each bloom layer
        this.composers.forEach((composer, layer) => {
            this.camera.layers.set(layer);
            composer.render();
        });
        
        // Reset camera layers and render final composition
        this.camera.layers.set(LAYERS.NORMAL_LAYER);
        this.finalComposer.render();
    }
    
    handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Resize all composers
        this.composers.forEach(composer => {
            composer.setSize(width, height);
        });
        
        if (this.finalComposer) {
            this.finalComposer.setSize(width, height);
        }
    }
    
    updateBloom(settings) {
        this.composers.forEach((composer, layer) => {
            composer.passes.forEach(pass => {
                if (pass instanceof UnrealBloomPass) {
                    switch (layer) {
                        case LAYERS.BLOOM:
                            pass.strength = (settings.nodeBloomStrength ?? pass.strength) * 1.2;
                            pass.radius = settings.nodeBloomRadius ?? pass.radius;
                            pass.threshold = (settings.nodeBloomThreshold ?? pass.threshold) * 0.8;
                            break;
                        case LAYERS.HOLOGRAM:
                            pass.strength = (settings.environmentBloomStrength ?? pass.strength) * 1.5;
                            pass.radius = (settings.environmentBloomRadius ?? pass.radius) * 1.2;
                            pass.threshold = (settings.environmentBloomThreshold ?? pass.threshold) * 0.7;
                            break;
                        case LAYERS.EDGE:
                            pass.strength = (settings.edgeBloomStrength ?? pass.strength) * 1.3;
                            pass.radius = settings.edgeBloomRadius ?? pass.radius;
                            pass.threshold = (settings.edgeBloomThreshold ?? pass.threshold) * 0.9;
                            break;
                    }
                }
            });
        });
    }
    
    handleSettingsUpdate(event) {
        const settings = event.detail;
        
        if (settings.bloom) {
            this.bloomSettings = settings.bloom;
            this.updateBloom(settings.bloom);
        }

        if (settings.hologram) {
            this.hologramSettings = settings.hologram;
            this.createHologramStructure();
        }
    }
    
    dispose() {
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);
        
        // Dispose all composers
        this.composers.forEach(composer => {
            composer.dispose();
        });
        
        if (this.finalComposer) {
            this.finalComposer.dispose();
        }

        // Dispose hologram resources
        this.hologramGroup.children.forEach(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
        this.scene.remove(this.hologramGroup);
    }
}
