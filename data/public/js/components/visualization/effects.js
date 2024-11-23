import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { BLOOM_LAYER, NORMAL_LAYER } from './nodes.js';
import { visualizationSettings } from '../../services/visualizationSettings.js';

export class EffectsManager {
    constructor(scene, camera, renderer, settings = {}) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        this.bloomComposer = null;
        this.finalComposer = null;
        this.renderTarget = null;
        
        // Get bloom settings from visualization settings service
        const bloomSettings = visualizationSettings.getBloomSettings();
        this.bloomStrength = bloomSettings.nodeBloomStrength;
        this.bloomRadius = bloomSettings.nodeBloomRadius;
        this.bloomThreshold = bloomSettings.nodeBloomThreshold;

        // Hologram settings from visualization settings
        const hologramSettings = visualizationSettings.getHologramSettings();
        this.hologramGroup = new THREE.Group();
        this.scene.add(this.hologramGroup);
        this.hologramColor = new THREE.Color(hologramSettings.color);
        this.hologramScale = hologramSettings.scale;
        this.hologramOpacity = hologramSettings.opacity;

        // Fisheye settings from visualization settings
        const fisheyeSettings = visualizationSettings.getFisheyeSettings();
        this.fisheyeEnabled = fisheyeSettings.enabled;
        this.fisheyeStrength = fisheyeSettings.strength;
        this.fisheyeRadius = fisheyeSettings.radius;
        this.fisheyeFocusPoint = [fisheyeSettings.focusX, fisheyeSettings.focusY, fisheyeSettings.focusZ];

        // Bind the settings update handler
        this.handleSettingsUpdate = this.handleSettingsUpdate.bind(this);
        window.addEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

        // Initialize post-processing after a short delay to ensure renderer is ready
        requestAnimationFrame(() => this.initPostProcessing());
    }

    handleSettingsUpdate(event) {
        const settings = event.detail;
            
        // Update bloom settings if they've changed
        if (settings.nodeBloomStrength !== undefined ||
            settings.nodeBloomRadius !== undefined ||
            settings.nodeBloomThreshold !== undefined) {
            const bloomSettings = visualizationSettings.getBloomSettings();
            this.updateBloom(bloomSettings);
        }

        // Update hologram settings if they've changed
        if (settings.hologramColor !== undefined ||
            settings.hologramScale !== undefined ||
            settings.hologramOpacity !== undefined) {
            const hologramSettings = visualizationSettings.getHologramSettings();
            this.updateFeature('hologramColor', hologramSettings.color);
            this.updateFeature('hologramScale', hologramSettings.scale);
            this.updateFeature('hologramOpacity', hologramSettings.opacity);
        }

        // Update fisheye settings if they've changed
        if (settings.fisheye) {
            const fisheyeSettings = visualizationSettings.getFisheyeSettings();
            this.updateFisheye(fisheyeSettings);
        }
    }

    createRenderTarget() {
        if (this.renderTarget) {
            this.renderTarget.dispose();
        }
        
        this.renderTarget = new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                type: THREE.HalfFloatType
            }
        );
        return this.renderTarget;
    }

    initPostProcessing() {
        if (!this.renderer || !this.renderer.domElement) {
            console.warn('Renderer not ready, deferring post-processing initialization');
            return;
        }

        // Create render targets
        const renderTarget = this.createRenderTarget();

        // Setup bloom composer
        this.bloomComposer = new EffectComposer(this.renderer, renderTarget);
        this.bloomComposer.renderToScreen = false;
        
        const renderScene = new RenderPass(this.scene, this.camera);
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            this.bloomStrength,
            this.bloomRadius,
            this.bloomThreshold
        );

        this.bloomComposer.addPass(renderScene);
        this.bloomComposer.addPass(bloomPass);

        // Setup final composer
        this.finalComposer = new EffectComposer(this.renderer);
        this.finalComposer.addPass(renderScene);

        // Add custom shader pass to combine bloom with scene
        const finalPass = new ShaderPass(
            new THREE.ShaderMaterial({
                uniforms: {
                    baseTexture: { value: null },
                    bloomTexture: { value: null }
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
                    uniform sampler2D bloomTexture;
                    varying vec2 vUv;
                    void main() {
                        vec4 baseColor = texture2D(baseTexture, vUv);
                        vec4 bloomColor = texture2D(bloomTexture, vUv);
                        gl_FragColor = baseColor + bloomColor;
                    }
                `,
                defines: {}
            }),
            "baseTexture"
        );
        finalPass.needsSwap = true;
        
        // Update bloom texture reference after render
        this.bloomComposer.renderToScreen = false;
        this.bloomComposer.onAfterRender = () => {
            if (finalPass.uniforms && this.bloomComposer.renderTarget2) {
                finalPass.uniforms.bloomTexture.value = this.bloomComposer.renderTarget2.texture;
            }
        };

        this.finalComposer.addPass(finalPass);
    }

    createHologramStructure() {
        // Clear existing hologram structure
        while (this.hologramGroup.children.length > 0) {
            const child = this.hologramGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.hologramGroup.remove(child);
        }

        // Get material settings
        const materialSettings = visualizationSettings.getNodeSettings().material;

        // Create multiple rings with different sizes to match each sphere's radius
        const ringSizes = [40, 30, 20];
        for (let i = 0; i < 3; i++) {
            const ringGeometry = new THREE.TorusGeometry(ringSizes[i], 3, 16, 100);
            const ringMaterial = new THREE.MeshStandardMaterial({
                color: this.hologramColor,
                emissive: this.hologramColor,
                emissiveIntensity: materialSettings.emissiveMaxIntensity,
                transparent: true,
                opacity: this.hologramOpacity,
                metalness: materialSettings.metalness,
                roughness: materialSettings.roughness
            });

            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2 * i;
            ring.rotation.y = Math.PI / 4 * i;
            ring.userData.rotationSpeed = 0.002 * (i + 1);
            ring.layers.enable(BLOOM_LAYER);
            this.hologramGroup.add(ring);
        }

        // Add Buckminster Fullerene 
        const buckyGeometry = new THREE.IcosahedronGeometry(40 * this.hologramScale, 1);
        const buckyMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const buckySphere = new THREE.Mesh(buckyGeometry, buckyMaterial);
        buckySphere.userData.rotationSpeed = 0.0001;
        buckySphere.layers.enable(BLOOM_LAYER);
        this.hologramGroup.add(buckySphere);

        // Add Geodesic Dome
        const geodesicGeometry = new THREE.IcosahedronGeometry(30 * this.hologramScale, 1);
        const geodesicMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const geodesicDome = new THREE.Mesh(geodesicGeometry, geodesicMaterial);
        geodesicDome.userData.rotationSpeed = 0.0002;
        geodesicDome.layers.enable(BLOOM_LAYER);
        this.hologramGroup.add(geodesicDome);

        // Add Normal Triangle Sphere
        const triangleGeometry = new THREE.SphereGeometry(20 * this.hologramScale, 32, 32);
        const triangleMaterial = new THREE.MeshBasicMaterial({
            color: this.hologramColor,
            wireframe: true,
            transparent: true,
            opacity: this.hologramOpacity
        });
        const triangleSphere = new THREE.Mesh(triangleGeometry, triangleMaterial);
        triangleSphere.userData.rotationSpeed = 0.0003;
        triangleSphere.layers.enable(BLOOM_LAYER);
        this.hologramGroup.add(triangleSphere);
    }

    animate() {
        // Animate all hologram elements
        this.hologramGroup.children.forEach(child => {
            child.rotation.x += child.userData.rotationSpeed;
            child.rotation.y += child.userData.rotationSpeed;
        });
    }

    render() {
        if (!this.bloomComposer || !this.finalComposer) {
            return;
        }

        // Store original layer state
        const originalLayers = this.camera.layers.mask;

        // First render the bloom layer
        this.camera.layers.set(BLOOM_LAYER);
        this.bloomComposer.render();
        
        // Then render the normal scene
        this.camera.layers.set(NORMAL_LAYER);
        this.finalComposer.render();

        // Restore original layer state
        this.camera.layers.mask = originalLayers;
    }

    handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Recreate render targets with new size
        const renderTarget = this.createRenderTarget();

        // Update composers
        if (this.bloomComposer) {
            this.bloomComposer.setSize(width, height);
            this.bloomComposer.renderTarget1.setSize(width, height);
            this.bloomComposer.renderTarget2.setSize(width, height);
        }
        
        if (this.finalComposer) {
            this.finalComposer.setSize(width, height);
            this.finalComposer.renderTarget1.setSize(width, height);
            this.finalComposer.renderTarget2.setSize(width, height);
        }
    }

    updateFeature(control, value) {
        console.log(`Updating effect feature: ${control} = ${value}`);
        switch (control) {
            // Bloom features
            case 'bloomStrength':
                this.bloomStrength = value;
                if (this.bloomComposer) {
                    this.bloomComposer.passes.forEach(pass => {
                        if (pass instanceof UnrealBloomPass) {
                            pass.strength = value;
                        }
                    });
                }
                break;
            case 'bloomRadius':
                this.bloomRadius = value;
                if (this.bloomComposer) {
                    this.bloomComposer.passes.forEach(pass => {
                        if (pass instanceof UnrealBloomPass) {
                            pass.radius = value;
                        }
                    });
                }
                break;
            case 'bloomThreshold':
                this.bloomThreshold = value;
                if (this.bloomComposer) {
                    this.bloomComposer.passes.forEach(pass => {
                        if (pass instanceof UnrealBloomPass) {
                            pass.threshold = value;
                        }
                    });
                }
                break;

            // Hologram features
            case 'hologramColor':
                if (typeof value === 'number' || typeof value === 'string') {
                    this.hologramColor = new THREE.Color(value);
                    this.hologramGroup.children.forEach(child => {
                        if (child.material) {
                            child.material.color.copy(this.hologramColor);
                            if (child.material.emissive) {
                                child.material.emissive.copy(this.hologramColor);
                            }
                        }
                    });
                }
                break;
            case 'hologramScale':
                this.hologramScale = value;
                this.hologramGroup.scale.setScalar(value);
                break;
            case 'hologramOpacity':
                this.hologramOpacity = value;
                this.hologramGroup.children.forEach(child => {
                    if (child.material) {
                        child.material.opacity = value;
                    }
                });
                break;
        }
    }

    updateBloom(settings) {
        console.log('Updating bloom settings:', settings);
        if (!this.bloomComposer) return;

        this.bloomComposer.passes.forEach(pass => {
            if (pass instanceof UnrealBloomPass) {
                if (settings.nodeBloomStrength !== undefined) {
                    pass.strength = settings.nodeBloomStrength;
                }
                if (settings.nodeBloomRadius !== undefined) {
                    pass.radius = settings.nodeBloomRadius;
                }
                if (settings.nodeBloomThreshold !== undefined) {
                    pass.threshold = settings.nodeBloomThreshold;
                }
            }
        });

        // Store the updated values
        this.bloomStrength = settings.nodeBloomStrength ?? this.bloomStrength;
        this.bloomRadius = settings.nodeBloomRadius ?? this.bloomRadius;
        this.bloomThreshold = settings.nodeBloomThreshold ?? this.bloomThreshold;
    }

    updateFisheye(settings) {
        console.log('Updating fisheye settings:', settings);
        this.fisheyeEnabled = settings.enabled;
        this.fisheyeStrength = settings.strength;
        this.fisheyeRadius = settings.radius;
        this.fisheyeFocusPoint = settings.focusPoint;

        // Apply fisheye effect if enabled
        if (this.fisheyeEnabled) {
            // TODO: Implement fisheye distortion shader
            console.log('Fisheye effect enabled:', {
                strength: this.fisheyeStrength,
                radius: this.fisheyeRadius,
                focusPoint: this.fisheyeFocusPoint
            });
        }
    }

    dispose() {
        // Remove event listener
        window.removeEventListener('visualizationSettingsUpdated', this.handleSettingsUpdate);

        // Dispose render targets
        if (this.renderTarget) {
            this.renderTarget.dispose();
        }

        // Dispose bloom resources
        if (this.bloomComposer) {
            this.bloomComposer.renderTarget1.dispose();
            this.bloomComposer.renderTarget2.dispose();
        }
        if (this.finalComposer) {
            this.finalComposer.renderTarget1.dispose();
            this.finalComposer.renderTarget2.dispose();
        }

        // Dispose hologram resources
        this.hologramGroup.children.forEach(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    }
}
