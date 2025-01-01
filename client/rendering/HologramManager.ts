import {
    Scene,
    Group,
    Mesh,
    Vector3,
    WebGLRenderer
} from 'three';
import { Settings } from '../types/settings';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { HologramShaderMaterial } from './materials/HologramShaderMaterial';

export class HologramManager {
    private readonly group = new Group();
    private isXRMode = false;
    private readonly geometryFactory: GeometryFactory;
    private readonly materialFactory: MaterialFactory;

    constructor(
        private readonly scene: Scene,
        _renderer: WebGLRenderer,  // Used by subclasses
        private settings: Settings
    ) {
        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        this.createHolograms();
        this.scene.add(this.group);
    }

    private createHologramMesh(type: string, quality: string = 'medium'): Mesh {
        const geometry = this.geometryFactory.getHologramGeometry(type as 'ring' | 'sphere' | 'icosahedron', quality as 'low' | 'medium' | 'high');
        const material = this.materialFactory.createHologramMaterial();
        return new Mesh(geometry, material);
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

        const quality = this.isXRMode ? 'high' : this.settings.xr.quality;
        for (let i = 0; i < this.settings.visualization.hologram.ringCount; i++) {
            const ring = this.createHologramMesh('ring', quality);
            const scale = this.settings.visualization.hologram.ringSizes[i] || 20;
            ring.scale.set(scale, scale, scale);
            ring.rotateX(Math.PI / 2 * i);
            ring.rotateY(Math.PI / 4 * i);
            ring.userData.rotationSpeed = this.settings.visualization.hologram.ringRotationSpeed * (i + 1);
            this.group.add(ring);
        }

        if (this.settings.visualization.hologram.enableBuckminster) {
            const mesh = this.createHologramMesh('buckminster', quality);
            const scale = this.settings.visualization.hologram.buckminsterScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.visualization.hologram.buckminsterOpacity;
            this.group.add(mesh);
        }

        if (this.settings.visualization.hologram.enableGeodesic) {
            const mesh = this.createHologramMesh('geodesic', quality);
            const scale = this.settings.visualization.hologram.geodesicScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.visualization.hologram.geodesicOpacity;
            this.group.add(mesh);
        }

        if (this.settings.visualization.hologram.enableTriangleSphere) {
            const mesh = this.createHologramMesh('triangleSphere', quality);
            const scale = this.settings.visualization.hologram.triangleSphereScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.visualization.hologram.triangleSphereOpacity;
            this.group.add(mesh);
        }
    }

    setXRMode(enabled: boolean) {
        this.isXRMode = enabled;
        this.group.traverse(child => {
            if (child instanceof Mesh && child.material instanceof HologramShaderMaterial) {
                child.material.defines = { USE_AR: '' };
                child.material.needsUpdate = true;
            }
        });
        this.createHolograms();
    }

    handleInteraction(position: Vector3) {
        this.group.traverse(child => {
            if (child instanceof Mesh && child.material instanceof HologramShaderMaterial) {
                const distance = position.distanceTo(child.position);
                if (distance < 0.5) {
                    child.material.uniforms.pulseIntensity.value = 0.4;
                    setTimeout(() => {
                        if (child.material instanceof HologramShaderMaterial) {
                            child.material.uniforms.pulseIntensity.value = 0.2;
                        }
                    }, 500);
                }
            }
        });
    }

    update(deltaTime: number) {
        this.group.traverse(child => {
            if (child instanceof Mesh) {
                child.rotateY((child.userData.rotationSpeed || this.settings.visualization.hologram.globalRotationSpeed) * deltaTime);
                if (child.material instanceof HologramShaderMaterial) {
                    child.material.uniforms.time.value += deltaTime;
                }
            }
        });
    }

    updateSettings(newSettings: Settings) {
        this.settings = newSettings;
        this.materialFactory.updateMaterial('hologram', this.settings);
        this.createHolograms();
    }

    getGroup() {
        return this.group;
    }

    dispose() {
        // Geometries and materials are managed by the factories
        this.scene.remove(this.group);
    }
}
