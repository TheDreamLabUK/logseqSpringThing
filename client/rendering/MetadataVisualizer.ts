import * as THREE from 'three';
import { Settings } from '../core/types';

export class MetadataVisualizer {
    private readonly geometries = {
        complex: new THREE.IcosahedronGeometry(1, 2),
        medium: new THREE.SphereGeometry(1, 16, 12),
        simple: new THREE.BoxGeometry(1, 1, 1)
    };

    private readonly material = new THREE.MeshBasicMaterial({
        color: new THREE.Color('#00ff00'),
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide
    });

    constructor(
        private readonly camera: THREE.PerspectiveCamera,
        private readonly settings: Settings
    ) {}

    createNodeMesh(metadata: { name: string; commitAge: number; hyperlinkCount: number; importance: number; position: THREE.Vector3 }) {
        const geometry = this.selectGeometry(metadata.importance);
        const mesh = new THREE.Mesh(geometry, this.createNodeMaterial(metadata));

        mesh.scale.set(
            this.settings.nodes.sizeRange[0],
            this.settings.nodes.sizeRange[0],
            this.settings.nodes.sizeRange[0]
        );

        mesh.position.copy(metadata.position);

        this.billboardUpdate(mesh);

        return mesh;
    }

    private selectGeometry(importance: number): THREE.BufferGeometry {
        if (importance > 0.7) return this.geometries.complex;
        if (importance > 0.3) return this.geometries.medium;
        return this.geometries.simple;
    }

    private createNodeMaterial(metadata: { name: string; commitAge: number; hyperlinkCount: number }): THREE.Material {
        const ageColor = new THREE.Color(this.settings.nodes.colorRangeAge[0]);
        const linkColor = new THREE.Color(this.settings.nodes.colorRangeLinks[0]);
        const linkInfluence = Math.min(metadata.hyperlinkCount / 10, 1); // 10 links max

        const finalColor = new THREE.Color().copy(ageColor).lerp(linkColor, linkInfluence);

        const material = new THREE.MeshBasicMaterial({
            color: finalColor,
            transparent: true,
            opacity: this.settings.nodes.opacity
        });

        return material;
    }

    private billboardUpdate(mesh: THREE.Mesh) {
        const updateQuaternion = () => {
            mesh.quaternion.set(
                this.camera.quaternion.x,
                this.camera.quaternion.y,
                this.camera.quaternion.z,
                this.camera.quaternion.w
            );
        };
        (mesh as any).onBeforeRender = updateQuaternion;
    }

    dispose() {
        Object.values(this.geometries).forEach(geometry => geometry.dispose());
        this.material.dispose();
    }
}
