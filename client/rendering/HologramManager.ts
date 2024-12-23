import {
    Scene,
    Group,
    Mesh,
    Vector3,
    WebGLRenderer
} from 'three';
import { Settings } from '../core/types';
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

    private createHolograms() {
        while (this.group.children.length > 0) {
            const child = this.group.children[0];
            this.group.remove(child);
            if (child instanceof Mesh) {
                child.geometry.dispose();
                child.material.dispose();
            }
        }

        const quality = this.isXRMode ? this.settings.hologram.xrQuality : this.settings.hologram.desktopQuality;
        const material = this.materialFactory.getHologramMaterial(this.settings);

        for (let i = 0; i < this.settings.hologram.ringCount; i++) {
            const ring = new Mesh(
                this.geometryFactory.getHologramGeometry('ring', quality),
                material.clone()
            );
            const scale = this.settings.hologram.ringSizes[i] || 20;
            ring.scale.set(scale, scale, scale);
            ring.rotateX(Math.PI / 2 * i);
            ring.rotateY(Math.PI / 4 * i);
            ring.userData.rotationSpeed = this.settings.hologram.ringRotationSpeed * (i + 1);
            this.group.add(ring);
        }

        if (this.settings.hologram.enableBuckminster) {
            const mesh = new Mesh(
                this.geometryFactory.getHologramGeometry('buckminster', quality),
                material.clone()
            );
            const scale = this.settings.hologram.buckminsterScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.hologram.buckminsterOpacity;
            this.group.add(mesh);
        }

        if (this.settings.hologram.enableGeodesic) {
            const mesh = new Mesh(
                this.geometryFactory.getHologramGeometry('geodesic', quality),
                material.clone()
            );
            const scale = this.settings.hologram.geodesicScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.hologram.geodesicOpacity;
            this.group.add(mesh);
        }

        if (this.settings.hologram.enableTriangleSphere) {
            const mesh = new Mesh(
                this.geometryFactory.getHologramGeometry('triangleSphere', quality),
                material.clone()
            );
            const scale = this.settings.hologram.triangleSphereScale;
            mesh.scale.set(scale, scale, scale);
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.hologram.triangleSphereOpacity;
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
                child.rotateY((child.userData.rotationSpeed || this.settings.hologram.globalRotationSpeed) * deltaTime);
                if (child.material instanceof HologramShaderMaterial) {
                    child.material.uniforms.time.value += deltaTime;
                }
            }
        });
    }

    updateSettings(newSettings: Partial<Settings>) {
        Object.assign(this.settings, newSettings);
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
