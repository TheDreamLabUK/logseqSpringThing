import {
    Scene,
    Group,
    Mesh,
    Vector3,
    WebGLRenderer
} from 'three';
import { Settings } from '../types/settings';
import { GeometryFactory } from '../rendering/factories/GeometryFactory';
import { MaterialFactory } from '../rendering/factories/MaterialFactory';
import { HologramShaderMaterial } from '../rendering/materials/HologramShaderMaterial';

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

        const quality = this.isXRMode ? 'high' : this.settings.xr.quality;
        const material = this.materialFactory.getHologramMaterial(this.settings);

        // Create rings using native world units (40, 80, 120)
        const sphereSizes = this.settings.visualization.hologram.sphereSizes;
        for (let i = 0; i < this.settings.visualization.hologram.ringCount; i++) {
            const size = sphereSizes[i] || 40 * (i + 1); // Default to 40, 80, 120 pattern if not specified
            const ring = new Mesh(
                this.geometryFactory.getHologramGeometry('ring', quality, size),
                material.clone()
            );
            // Position rings at different angles to make them more visible
            ring.rotateX(Math.PI / 3 * i);  // Changed from PI/2 to PI/3
            ring.rotateY(Math.PI / 6 * i);  // Changed from PI/4 to PI/6
            ring.userData.rotationSpeed = this.settings.visualization.hologram.ringRotationSpeed * (i + 1);
            (ring.material as HologramShaderMaterial).setEdgeOnly(true);
            this.group.add(ring);
        }

        if (this.settings.visualization.hologram.enableBuckminster) {
            const size = this.settings.visualization.hologram.buckminsterSize;
            const mesh = new Mesh(
                this.geometryFactory.getHologramGeometry('buckminster', quality, size),
                material.clone()
            );
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.visualization.hologram.buckminsterOpacity;
            (mesh.material as HologramShaderMaterial).setEdgeOnly(true);
            this.group.add(mesh);
        }

        if (this.settings.visualization.hologram.enableGeodesic) {
            const size = this.settings.visualization.hologram.geodesicSize;
            const mesh = new Mesh(
                this.geometryFactory.getHologramGeometry('geodesic', quality, size),
                material.clone()
            );
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.visualization.hologram.geodesicOpacity;
            (mesh.material as HologramShaderMaterial).setEdgeOnly(true);
            this.group.add(mesh);
        }

        if (this.settings.visualization.hologram.enableTriangleSphere) {
            const size = this.settings.visualization.hologram.triangleSphereSize;
            const mesh = new Mesh(
                this.geometryFactory.getHologramGeometry('triangleSphere', quality, size),
                material.clone()
            );
            (mesh.material as HologramShaderMaterial).uniforms.opacity.value = this.settings.visualization.hologram.triangleSphereOpacity;
            (mesh.material as HologramShaderMaterial).setEdgeOnly(true);
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
                if (distance < 0.5 && child.material.uniforms) {
                    child.material.uniforms.pulseIntensity.value = 0.4;
                    setTimeout(() => {
                        if (child.material instanceof HologramShaderMaterial && child.material.uniforms) {
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
                if (child.material instanceof HologramShaderMaterial && child.material.uniforms) {
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
