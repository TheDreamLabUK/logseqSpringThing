import * as THREE from 'three';
import { HologramSettings } from '../types/metadata';
import { XRHandedness } from '../types/xr';

export class HologramManager {
    private readonly hologramGroup: THREE.Group;
    private readonly geometryCache: Map<string, THREE.BufferGeometry>;
    private readonly materialCache: Map<string, THREE.Material>;
    private isXRMode: boolean = false;

    constructor(
        private readonly scene: THREE.Scene,
        private readonly camera: THREE.PerspectiveCamera,
        private settings: HologramSettings
    ) {
        this.hologramGroup = new THREE.Group();
        this.geometryCache = new Map();
        this.materialCache = new Map();
        this.scene.add(this.hologramGroup);
        
        // Initialize geometries based on quality setting
        this.initializeGeometries();
        this.createHolographicStructures();
    }

    private initializeGeometries(): void {
        const quality = this.isXRMode ? this.settings.xrQuality : this.settings.desktopQuality;
        const segments = {
            low: { ring: 32, sphere: 8 },
            medium: { ring: 64, sphere: 16 },
            high: { ring: 128, sphere: 32 }
        }[quality] || segments.medium;

        // Create and cache geometries
        this.geometryCache.set('ring', new THREE.TorusGeometry(1, 0.02, segments.ring, segments.ring * 2));
        this.geometryCache.set('buckminster', new THREE.IcosahedronGeometry(1, quality === 'high' ? 2 : 1));
        this.geometryCache.set('geodesic', new THREE.IcosahedronGeometry(1, quality === 'low' ? 1 : 2));
        this.geometryCache.set('triangleSphere', new THREE.SphereGeometry(1, segments.sphere, segments.sphere));

        // Create hologram material
        const hologramMaterial = new THREE.ShaderMaterial({
            uniforms: {
                color: { value: new THREE.Color(this.settings.ringColor) },
                opacity: { value: this.settings.ringOpacity },
                time: { value: 0 },
                pulseSpeed: { value: 1.0 },
                pulseIntensity: { value: 0.2 }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec2 vUv;
                void main() {
                    vPosition = position;
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform float opacity;
                uniform float time;
                uniform float pulseSpeed;
                uniform float pulseIntensity;
                varying vec3 vPosition;
                varying vec2 vUv;
                
                void main() {
                    float pulse = sin(time * pulseSpeed) * pulseIntensity + 1.0;
                    float edge = 1.0 - smoothstep(0.4, 0.5, abs(vUv.y - 0.5));
                    vec3 finalColor = color * pulse;
                    float finalOpacity = opacity * edge;
                    
                    #ifdef USE_AR
                        // Adjust opacity based on distance to camera for AR
                        float depth = gl_FragCoord.z / gl_FragCoord.w;
                        finalOpacity *= smoothstep(10.0, 0.0, depth);
                    #endif
                    
                    gl_FragColor = vec4(finalColor, finalOpacity);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });

        this.materialCache.set('hologram', hologramMaterial);
    }

    private createHolographicStructures(): void {
        // Clear existing structures
        while (this.hologramGroup.children.length > 0) {
            const child = this.hologramGroup.children[0];
            this.hologramGroup.remove(child);
        }

        // Create rings
        const ringGeometry = this.geometryCache.get('ring')!;
        const material = this.materialCache.get('hologram')!.clone();

        for (let i = 0; i < this.settings.ringCount; i++) {
            const ring = new THREE.Mesh(ringGeometry, material.clone());
            ring.scale.setScalar(this.settings.ringSizes[i] || 20);
            ring.rotation.x = Math.PI / 2 * i;
            ring.rotation.y = Math.PI / 4 * i;
            ring.userData.rotationSpeed = this.settings.ringRotationSpeed * (i + 1);
            this.hologramGroup.add(ring);
        }

        // Create geometric structures based on settings
        if (this.settings.enableBuckminster) {
            const geometry = this.geometryCache.get('buckminster')!;
            const mesh = new THREE.Mesh(geometry, material.clone());
            mesh.scale.setScalar(this.settings.buckminsterScale);
            mesh.material.uniforms.opacity.value = this.settings.buckminsterOpacity;
            this.hologramGroup.add(mesh);
        }

        if (this.settings.enableGeodesic) {
            const geometry = this.geometryCache.get('geodesic')!;
            const mesh = new THREE.Mesh(geometry, material.clone());
            mesh.scale.setScalar(this.settings.geodesicScale);
            mesh.material.uniforms.opacity.value = this.settings.geodesicOpacity;
            this.hologramGroup.add(mesh);
        }

        if (this.settings.enableTriangleSphere) {
            const geometry = this.geometryCache.get('triangleSphere')!;
            const mesh = new THREE.Mesh(geometry, material.clone());
            mesh.scale.setScalar(this.settings.triangleSphereScale);
            mesh.material.uniforms.opacity.value = this.settings.triangleSphereOpacity;
            this.hologramGroup.add(mesh);
        }
    }

    public setXRMode(enabled: boolean): void {
        this.isXRMode = enabled;
        if (enabled) {
            // Optimize for XR
            this.hologramGroup.children.forEach(child => {
                if (child instanceof THREE.Mesh) {
                    child.material.defines = { USE_AR: '' };
                    child.material.needsUpdate = true;
                }
            });
        }
        // Recreate geometries with appropriate quality
        this.initializeGeometries();
        this.createHolographicStructures();
    }

    public update(deltaTime: number): void {
        // Update hologram animations
        this.hologramGroup.children.forEach(child => {
            if (child instanceof THREE.Mesh) {
                child.rotation.y += (child.userData.rotationSpeed || this.settings.globalRotationSpeed) * deltaTime;
                if (child.material instanceof THREE.ShaderMaterial) {
                    child.material.uniforms.time.value += deltaTime;
                }
            }
        });
    }

    public handleHandInteraction(hand: THREE.XRHand, handedness: XRHandedness): void {
        if (!this.isXRMode) return;

        // Get index finger tip position
        const indexTip = hand.joints['index-finger-tip'];
        if (!indexTip) return;

        // Convert joint position to world space
        const fingerPosition = new THREE.Vector3()
            .fromBufferAttribute(indexTip.position as THREE.BufferAttribute, 0)
            .applyMatrix4(hand.matrixWorld);

        // Check interaction with hologram elements
        this.hologramGroup.children.forEach(child => {
            if (child instanceof THREE.Mesh) {
                const distance = fingerPosition.distanceTo(child.position);
                if (distance < 0.1) { // 10cm interaction radius
                    // Increase pulse intensity temporarily
                    if (child.material instanceof THREE.ShaderMaterial) {
                        child.material.uniforms.pulseIntensity.value = 0.4;
                        setTimeout(() => {
                            child.material.uniforms.pulseIntensity.value = 0.2;
                        }, 500);
                    }
                }
            }
        });
    }

    public updateSettings(settings: Partial<HologramSettings>): void {
        Object.assign(this.settings, settings);
        this.createHolographicStructures();
    }

    public dispose(): void {
        // Dispose geometries
        this.geometryCache.forEach(geometry => geometry.dispose());
        this.geometryCache.clear();

        // Dispose materials
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();

        // Remove from scene
        this.scene.remove(this.hologramGroup);
    }
}
