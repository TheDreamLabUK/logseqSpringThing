import * as THREE from 'three';
import {
    Scene,
    PerspectiveCamera,
    Group,
    BufferGeometry,
    Material,
    Mesh,
    Vector3,
    DoubleSide,
    Color,
    TorusBufferGeometry,
    IcosahedronBufferGeometry,
    SphereBufferGeometry,
    ShaderMaterial
} from 'three';
import { XRHandWithHaptics } from '../xr/xrTypes';
import { Settings } from '../core/types';

interface Segments {
    ring: number;
    sphere: number;
}

export class HologramManager {
    private readonly geometryCache = new Map<string, BufferGeometry>();
    private readonly materialCache = new Map<string, ShaderMaterial>();
    private readonly group = new Group();
    private isXRMode = false;

    constructor(
        private readonly scene: Scene,
        private readonly camera: PerspectiveCamera,
        private settings: Settings
    ) {
        this.initGeometries();
        this.initMaterials();
        this.createHolograms();
        this.scene.add(this.group);
    }

    private initGeometries() {
        const quality = this.isXRMode ? this.settings.hologram.xrQuality : this.settings.hologram.desktopQuality;
        const segments: Segments = {
            low: { ring: 32, sphere: 8 },
            medium: { ring: 64, sphere: 16 },
            high: { ring: 128, sphere: 32 }
        }[quality] || { ring: 64, sphere: 16 };

        this.geometryCache.set('ring', new TorusBufferGeometry(1, 0.02, segments.ring, segments.ring * 2));
        this.geometryCache.set('buckminster', new IcosahedronBufferGeometry(1, quality === 'high' ? 2 : 1));
        this.geometryCache.set('geodesic', new IcosahedronBufferGeometry(1, quality === 'low' ? 1 : 2));
        this.geometryCache.set('triangleSphere', new SphereBufferGeometry(1, segments.sphere, segments.sphere));
    }

    private initMaterials() {
        const hologramMaterial = new ShaderMaterial({
            uniforms: {
                color: { value: new Color(this.settings.hologram.ringColor) },
                opacity: { value: this.settings.hologram.ringOpacity },
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
                        float depth = gl_FragCoord.z / gl_FragCoord.w;
                        finalOpacity *= smoothstep(10.0, 0.0, depth);
                    #endif
                    
                    gl_FragColor = vec4(finalColor, finalOpacity);
                }
            `,
            transparent: true,
            side: DoubleSide,
            depthWrite: false
        });

        this.materialCache.set('hologram', hologramMaterial);
    }

    private createHolograms() {
        while (this.group.children.length > 0) {
            const child = this.group.children[0];
            this.group.remove(child);
            if (child instanceof Mesh) {
                child.geometry.dispose();
                child.material.dispose();
            }
        }

        const material = this.materialCache.get('hologram')!.clone();
        const ringGeometry = this.geometryCache.get('ring')!;

        for (let i = 0; i < this.settings.hologram.ringCount; i++) {
            const ring = new Mesh(ringGeometry, material.clone());
            const scale = this.settings.hologram.ringSizes[i] || 20;
            ring.scale.set(scale, scale, scale);
            ring.rotateX(Math.PI / 2 * i);
            ring.rotateY(Math.PI / 4 * i);
            ring.userData.rotationSpeed = this.settings.hologram.ringRotationSpeed * (i + 1);
            this.group.add(ring);
        }

        if (this.settings.hologram.enableBuckminster) {
            const geometry = this.geometryCache.get('buckminster')!;
            const mesh = new Mesh(geometry, material.clone());
            const scale = this.settings.hologram.buckminsterScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as ShaderMaterial).uniforms.opacity.value = this.settings.hologram.buckminsterOpacity;
            this.group.add(mesh);
        }

        if (this.settings.hologram.enableGeodesic) {
            const geometry = this.geometryCache.get('geodesic')!;
            const mesh = new Mesh(geometry, material.clone());
            const scale = this.settings.hologram.geodesicScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as ShaderMaterial).uniforms.opacity.value = this.settings.hologram.geodesicOpacity;
            this.group.add(mesh);
        }

        if (this.settings.hologram.enableTriangleSphere) {
            const geometry = this.geometryCache.get('triangleSphere')!;
            const mesh = new Mesh(geometry, material.clone());
            const scale = this.settings.hologram.triangleSphereScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as ShaderMaterial).uniforms.opacity.value = this.settings.hologram.triangleSphereOpacity;
            this.group.add(mesh);
        }
    }

    setXRMode(enabled: boolean) {
        this.isXRMode = enabled;
        this.group.traverse(child => {
            if (child instanceof Mesh && child.material instanceof ShaderMaterial) {
                child.material.defines = { USE_AR: '' };
                child.material.needsUpdate = true;
            }
        });
        // Recreate geometries with appropriate quality
        this.initGeometries();
        this.createHolograms();
    }

    handleInteraction(position: Vector3) {
        this.group.traverse(child => {
            if (child instanceof Mesh && child.material instanceof ShaderMaterial) {
                const distance = position.distanceTo(child.position);
                if (distance < 0.5) {
                    child.material.uniforms.pulseIntensity.value = 0.4;
                    setTimeout(() => {
                        if (child.material instanceof ShaderMaterial) {
                            child.material.uniforms.pulseIntensity.value = 0.2;
                        }
                    }, 500);
                }
            }
        });
    }

    handleHandInteraction(hand: XRHandWithHaptics) {
        if (!this.isXRMode) return;

        const indexTip = hand.hand.joints['index-finger-tip'];
        if (!indexTip) return;

        const position = new Vector3();
        position.setFromMatrixPosition(indexTip.matrixWorld);
        this.handleInteraction(position);
    }

    update(deltaTime: number) {
        this.group.traverse(child => {
            if (child instanceof Mesh) {
                child.rotateY((child.userData.rotationSpeed || this.settings.hologram.globalRotationSpeed) * deltaTime);
                if (child.material instanceof ShaderMaterial) {
                    child.material.uniforms.time.value += deltaTime;
                }
            }
        });
    }

    updateSettings(newSettings: Partial<Settings>) {
        Object.assign(this.settings, newSettings);
        this.createHolograms();
    }

    getGroup() {
        return this.group;
    }

    dispose() {
        // Dispose geometries
        this.geometryCache.forEach(geometry => geometry.dispose());
        this.geometryCache.clear();

        // Dispose materials
        this.materialCache.forEach(material => material.dispose());
        this.materialCache.clear();

        // Remove from scene
        this.scene.remove(this.group);
    }
}
