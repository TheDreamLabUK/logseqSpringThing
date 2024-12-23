import * as THREE from 'three';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { FontLoader, Font } from 'three/examples/jsm/loaders/FontLoader.js';
import { NodeMetadata } from '../types/metadata';

export class MetadataVisualizer {
    private readonly geometries = {
        SPHERE: new THREE.SphereGeometry(1, 32, 32),
        ICOSAHEDRON: new THREE.IcosahedronGeometry(1),
        OCTAHEDRON: new THREE.OctahedronGeometry(1)
    };

    private font: Font | null = null;
    private fontLoader: FontLoader;
    private readonly fontPath = '/fonts/helvetiker_regular.typeface.json';
    private readonly labelScale = 0.1;
    private readonly labelHeight = 0.1;
    private readonly labelGroup: THREE.Group;

    constructor(
        private readonly camera: THREE.Camera,
        private readonly settings: any
    ) {
        this.fontLoader = new FontLoader();
        this.loadFont();
        this.labelGroup = new THREE.Group();
    }

    private async loadFont(): Promise<void> {
        try {
            this.font = await this.fontLoader.loadAsync(this.fontPath);
        } catch (error) {
            console.error('Failed to load font:', error);
        }
    }

    public async createTextMesh(text: string): Promise<THREE.Mesh | null> {
        if (!this.font) {
            console.warn('Font not loaded yet');
            return null;
        }

        const geometry = new TextGeometry(text, {
            font: this.font,
            size: 1,
            height: this.labelHeight,
            curveSegments: 4,
            bevelEnabled: false
        });

        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.set(this.labelScale, this.labelScale, this.labelScale);

        // Center the text
        geometry.computeBoundingBox();
        const textWidth = geometry.boundingBox!.max.x - geometry.boundingBox!.min.x;
        mesh.position.x = -textWidth * this.labelScale / 2;

        return mesh;
    }

    public createNodeVisual(metadata: NodeMetadata): THREE.Mesh {
        const geometry = this.getGeometryFromAge(metadata.commitAge);
        const material = this.createMaterialFromHyperlinks(metadata.hyperlinkCount);
        const mesh = new THREE.Mesh(geometry, material);

        const scale = this.calculateScale(metadata.importance);
        mesh.scale.set(scale, scale, scale);

        mesh.position.set(
            metadata.position.x,
            metadata.position.y,
            metadata.position.z
        );

        return mesh;
    }

    private getGeometryFromAge(age: number): THREE.BufferGeometry {
        if (age < 7) return this.geometries.SPHERE;
        if (age < 30) return this.geometries.ICOSAHEDRON;
        return this.geometries.OCTAHEDRON;
    }

    private createMaterialFromHyperlinks(count: number): THREE.Material {
        const hue = Math.min(count / 10, 1) * 0.3; // 0 to 0.3 range
        const color = new THREE.Color().setHSL(hue, 0.7, 0.5);

        return new THREE.MeshPhongMaterial({
            color: color,
            shininess: 30,
            transparent: true,
            opacity: 0.9
        });
    }

    private calculateScale(importance: number): number {
        const [min, max] = this.settings.nodes.sizeRange;
        return min + (max - min) * Math.min(importance, 1);
    }

    public async createMetadataLabel(metadata: NodeMetadata): Promise<THREE.Group> {
        const group = new THREE.Group();

        // Create text for name
        const nameMesh = await this.createTextMesh(metadata.name);
        if (nameMesh) {
            nameMesh.position.y = 1.2;
            group.add(nameMesh);
        }

        // Create text for commit age
        const ageMesh = await this.createTextMesh(`${Math.round(metadata.commitAge)} days`);
        if (ageMesh) {
            ageMesh.position.y = 0.8;
            group.add(ageMesh);
        }

        // Create text for hyperlink count
        const linksMesh = await this.createTextMesh(`${metadata.hyperlinkCount} links`);
        if (linksMesh) {
            linksMesh.position.y = 0.4;
            group.add(linksMesh);
        }

        // Billboard behavior
        if (this.settings.labels?.billboard_mode === 'camera') {
            group.onBeforeRender = () => {
                group.quaternion.copy(this.camera.quaternion);
            };
        } else {
            // Vertical billboard - only rotate around Y
            group.onBeforeRender = () => {
                const cameraPos = this.camera.position.clone();
                cameraPos.y = group.position.y;
                group.lookAt(cameraPos);
            };
        }

        return group;
    }

    public dispose(): void {
        // Clean up geometries
        Object.values(this.geometries).forEach(geometry => geometry.dispose());
        
        // Clean up label group
        this.labelGroup.traverse(child => {
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
        });
    }
}
