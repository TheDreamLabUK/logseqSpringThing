import * as THREE from 'three';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { FontLoader, Font } from 'three/examples/jsm/loaders/FontLoader.js';
import { NodeMetadata } from '../types/metadata';

type GeometryWithBoundingBox = THREE.BufferGeometry & {
    boundingBox: THREE.Box3 | null;
    computeBoundingBox: () => void;
};

export class MetadataVisualizer {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private fontLoader: FontLoader;
    private font: Font | null;
    private fontPath: string;
    private labelGroup: THREE.Group;
    private settings: any;

    constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera, settings: any) {
        this.scene = scene;
        this.camera = camera;
        this.fontLoader = new FontLoader();
        this.font = null;
        this.fontPath = '/fonts/helvetiker_regular.typeface.json';
        this.labelGroup = new THREE.Group();
        this.settings = settings;
        this.scene.add(this.labelGroup);
        this.loadFont();
    }

    private readonly geometries = {
        SPHERE: new THREE.SphereGeometry(1, 32, 32),
        ICOSAHEDRON: new THREE.IcosahedronGeometry(1),
        OCTAHEDRON: new THREE.OctahedronGeometry(1)
    };

    private async loadFont(): Promise<void> {
        try {
            this.font = await new Promise((resolve, reject) => {
                this.fontLoader.load(this.fontPath, resolve, undefined, reject);
            });
        } catch (error) {
            console.error('Failed to load font:', error);
        }
    }

    public createLabel(text: string, position: THREE.Vector3): void {
        if (!this.font) {
            console.warn('Font not loaded yet');
            return;
        }

        const textGeometry = new TextGeometry(text, {
            font: this.font,
            size: this.settings.labelSize || 0.1,
            height: this.settings.labelHeight || 0.01
        });

        const material = new THREE.MeshBasicMaterial({
            color: this.settings.labelColor || 0xffffff
        });

        // Create mesh with the text geometry and center it
        const geometry = textGeometry as unknown as GeometryWithBoundingBox;
        geometry.computeBoundingBox();
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(position);

        if (geometry.boundingBox) {
            const width = geometry.boundingBox.max.x - geometry.boundingBox.min.x;
            mesh.position.x -= width / 2;
        }
        
        this.labelGroup.add(mesh);
    }

    public async createTextMesh(text: string): Promise<THREE.Mesh | null> {
        if (!this.font) {
            console.warn('Font not loaded yet');
            return null;
        }

        const textGeometry = new TextGeometry(text, {
            font: this.font,
            size: 1,
            height: 0.1, // Keep text thin for readability
            curveSegments: 4,
            bevelEnabled: false
        });

        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8
        });

        // Create mesh with the text geometry and center it
        const geometry = textGeometry as unknown as GeometryWithBoundingBox;
        geometry.computeBoundingBox();
        const mesh = new THREE.Mesh(geometry, material);

        if (geometry.boundingBox) {
            const width = geometry.boundingBox.max.x - geometry.boundingBox.min.x;
            mesh.position.x -= width / 2;
        }

        return mesh;
    }

    public createNodeVisual(metadata: NodeMetadata): THREE.Mesh {
        const geometry = this.getGeometryFromAge(metadata.commitAge);
        const material = this.createMaterialFromHyperlinks(metadata.hyperlinkCount);
        const mesh = new THREE.Mesh(geometry, material);

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

    public async createMetadataLabel(metadata: NodeMetadata): Promise<THREE.Group> {
        const group = new THREE.Group();

        // Create text for name
        const nameMesh = await this.createTextMesh(metadata.name);
        if (nameMesh) {
            nameMesh.position.y = 1.2;
            nameMesh.scale.setScalar(0.1); // Small scale for readability
            group.add(nameMesh);
        }

        // Create text for commit age
        const ageMesh = await this.createTextMesh(`${Math.round(metadata.commitAge)} days`);
        if (ageMesh) {
            ageMesh.position.y = 0.8;
            ageMesh.scale.setScalar(0.1); // Small scale for readability
            group.add(ageMesh);
        }

        // Create text for hyperlink count
        const linksMesh = await this.createTextMesh(`${metadata.hyperlinkCount} links`);
        if (linksMesh) {
            linksMesh.position.y = 0.4;
            linksMesh.scale.setScalar(0.1); // Small scale for readability
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
