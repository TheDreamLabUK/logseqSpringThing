import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { BLOOM_LAYER, NORMAL_LAYER } from './nodes.js';

export class EffectsManager {
    constructor(scene, camera, renderer, settings = {}) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        this.bloomComposer = null;
        this.finalComposer = null;
        
        // Bloom settings with defaults
        this.bloomStrength = settings.bloomStrength || 1.5;
        this.bloomRadius = settings.bloomRadius || 0.4;
        this.bloomThreshold = settings.bloomThreshold || 0.8;

        // Hologram settings with defaults
        this.hologramGroup = new THREE.Group();
        this.scene.add(this.hologramGroup);
        this.hologramColor = new THREE.Color(settings.hologramColor || 0xFFD700);
        this.hologramScale = settings.hologramScale || 1;
        this.hologramOpacity = settings.hologramOpacity || 0.1;

        // Fisheye settings with defaults
        this.fisheyeEnabled = false;
        this.fisheyeStrength = 0.5;
        this.fisheyeRadius = 100.0;
        this.fisheyeFocusPoint = [0, 0, 0];
    }

    initPostProcessing() {
        // Create render targets
        const renderTarget = new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                colorSpace: THREE.SRGBColorSpace
            }
        );

        // Setup bloom composer with settings
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
                    bloomTexture: { value: this.bloomComposer.renderTarget2.texture }
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
                        gl_FragColor = baseColor + vec4(1.0) * bloomColor;
                    }
                `,
                defines: {}
            }),
            "baseTexture"
        );
        finalPass.needsSwap = true;
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

        // Create multiple rings with different sizes to match each sphere's radius
        const ringSizes = [40, 30, 20];
        for (let i = 0; i < 3; i++) {
            const ringGeometry = new THREE.TorusGeometry(ringSizes[i], 3, 16, 100);
            const ringMaterial = new THREE.MeshStandardMaterial({
                color: this.hologramColor,
                emissive: this.hologramColor,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: this.hologramOpacity,
                metalness: 0.8,
                roughness: 0.2
            });

            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2 * i;
            ring.rotation.y = Math.PI / 4 * i;
            ring.userData.rotationSpeed = 0.002 * (i + 1);
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
        buckySphere.layers.enable(1);
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
        geodesicDome.layers.enable(1);
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
        triangleSphere.layers.enable(1);
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
        // Render with bloom effect
        this.camera.layers.set(BLOOM_LAYER);
        this.bloomComposer.render();
        
        this.camera.layers.set(NORMAL_LAYER);
        this.finalComposer.render();
    }

    handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        if (this.bloomComposer) this.bloomComposer.setSize(width, height);
        if (this.finalComposer) this.finalComposer.setSize(width, height);
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
