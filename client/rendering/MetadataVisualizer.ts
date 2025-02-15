import * as THREE from 'three';
import {
    Mesh,
    Group,
    MeshBasicMaterial,
    Vector3,
    DoubleSide, 
    BufferGeometry,
    Object3D
} from 'three';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { FontLoader, Font } from 'three/examples/jsm/loaders/FontLoader.js';
import { NodeMetadata } from '../types/metadata';
import { Settings } from '../types/settings';
import { platformManager } from '../platform/platformManager';

interface MetadataLabelGroup extends Group {
    name: string;
    userData: {
        isMetadata: boolean;
    };
}

export type MetadataLabelCallback = (group: MetadataLabelGroup) => void;

interface ExtendedTextGeometry extends TextGeometry {
    computeBoundingBox: () => void;
    boundingBox: {
        max: { x: number };
        min: { x: number };
    } | null;
}

export class MetadataVisualizer {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private fontLoader: FontLoader;
    private font: Font | null;
    private fontPath: string;
    private labelGroup: THREE.Group;
    private settings: Settings;
    private fontLoadAttempts: number = 0;

    constructor(camera: THREE.PerspectiveCamera, scene: THREE.Scene, settings: Settings) {
        this.camera = camera;
        this.scene = scene;
        this.fontLoader = new FontLoader();
        this.font = null;
        this.fontPath = '/fonts/helvetiker_regular.typeface.json';
        this.labelGroup = new THREE.Group();
        this.settings = settings;
        
        // Enable both layers by default for desktop mode
        this.labelGroup.layers.enable(0);
        this.labelGroup.layers.enable(1);
        
        this.scene.add(this.labelGroup);
        this.loadFont();
        
        // Set initial layer mode
        this.setXRMode(platformManager.isXRMode);
        
        // Listen for XR mode changes
        platformManager.on('xrmodechange', (enabled: boolean) => {
            this.setXRMode(enabled);
        });
    }

    private async loadFont(): Promise<void> {
        try {
            await this.attemptFontLoad();
        } catch (error) {
            console.error('Initial font load failed:', error);
            await this.retryFontLoad();
        }
    }

    private async attemptFontLoad(): Promise<void> {
        this.font = await new Promise((resolve, reject) => {
            this.fontLoader.load(
                this.fontPath,
                resolve,
                undefined,
                reject
            );
        });
    }

    private async retryFontLoad(maxAttempts: number = 3): Promise<void> {
        while (this.fontLoadAttempts < maxAttempts && !this.font) {
            this.fontLoadAttempts++;
            await new Promise(resolve => setTimeout(resolve, 1000));
            try {
                await this.attemptFontLoad();
                console.log('Font loaded successfully after retry');
                break;
            } catch (error) {
                console.error(`Font load attempt ${this.fontLoadAttempts} failed:`, error);
            }
        }
    }

    public async createMetadataLabel(metadata: NodeMetadata): Promise<MetadataLabelGroup> {
        const group = new Group() as MetadataLabelGroup;
        group.name = 'metadata-label';
        group.userData = { isMetadata: true };

        // Format file size
        const fileSizeFormatted = metadata.fileSize > 1024 * 1024 
            ? `${(metadata.fileSize / (1024 * 1024)).toFixed(1)}MB`
            : metadata.fileSize > 1024
                ? `${(metadata.fileSize / 1024).toFixed(1)}KB`
                : `${metadata.fileSize}B`;

        // Create text for file name and size
        const nameMesh = await this.createTextMesh(`${metadata.name} (${fileSizeFormatted})`);
        if (nameMesh) {
            nameMesh.position.y = 1.5;
            nameMesh.scale.setScalar(0.8);
            group.add(nameMesh);
        }

        // Create text for node size
        const nodeSizeMesh = await this.createTextMesh(`Size: ${metadata.nodeSize.toFixed(1)}`);
        if (nodeSizeMesh) {
            nodeSizeMesh.position.y = 1.0;
            nodeSizeMesh.scale.setScalar(0.7);
            group.add(nodeSizeMesh);
        }

        // Create text for hyperlink count
        const linksMesh = await this.createTextMesh(`${metadata.hyperlinkCount} links`);
        if (linksMesh) {
            linksMesh.position.y = 0.5;
            linksMesh.scale.setScalar(0.7);
            group.add(linksMesh);
        }

        // Center all text meshes horizontally
        group.children.forEach(child => {
            if (child instanceof Mesh) {
                const geometry = child.geometry as BufferGeometry;
                if (!geometry.boundingSphere) {
                    geometry.computeBoundingSphere();
                }
                if (geometry.boundingSphere) {
                    child.position.x = -geometry.boundingSphere.radius;
                }
            }
        });

        // Set up billboarding
        const tempVec = new Vector3();
        const billboardMode = this.settings.visualization.labels.billboardMode;

        const updateBillboard = () => {
            if (billboardMode === 'camera') {
                // Full billboard - always face camera
                group.quaternion.copy(this.camera.quaternion);
            } else {
                // Vertical billboard - only rotate around Y axis
                tempVec.copy(this.camera.position).sub(group.position);
                tempVec.y = 0;
                group.lookAt(tempVec.add(group.position));
            }
        };

        // Add to render loop
        group.onBeforeRender = updateBillboard;

        // Set initial layer
        this.setGroupLayer(group, platformManager.isXRMode);

        return group;
    }

    private async createTextMesh(text: string): Promise<Mesh | Group | null> {
        if (!this.font) {
            console.warn('Font not loaded yet');
            return null;
        }

        const textGeometry = new TextGeometry(text, {
            font: this.font,
            size: this.settings.visualization.labels.desktopFontSize / 12 || 0.4,
            height: 0.01,
            curveSegments: Math.max(4, this.settings.visualization.labels.textResolution || 4),
            bevelEnabled: false
        }) as ExtendedTextGeometry;

        textGeometry.computeBoundingBox();

        const material = new MeshBasicMaterial({
            color: this.settings.visualization.labels.textColor || '#ffffff',
            transparent: true,
            opacity: 1.0,
            side: DoubleSide,
            depthWrite: true,
            depthTest: true
        });

        // Add outline for better visibility
        if (this.settings.visualization.labels.textOutlineWidth > 0) {
            const outlineMaterial = new MeshBasicMaterial({
                color: this.settings.visualization.labels.textOutlineColor || '#000000',
                side: DoubleSide
            });
            
            const outlineWidth = this.settings.visualization.labels.textOutlineWidth;
            const outlineGeometry = new TextGeometry(text, {
                font: this.font,
                size: this.settings.visualization.labels.desktopFontSize / 12 || 0.4,
                height: 0.01,
                curveSegments: Math.max(4, this.settings.visualization.labels.textResolution || 4),
                bevelEnabled: false
            }) as unknown as BufferGeometry;
            
            const outlineMesh = new Mesh(outlineGeometry, outlineMaterial);
            outlineMesh.scale.multiplyScalar(1 + outlineWidth);
            const textMesh = new Mesh(textGeometry as unknown as BufferGeometry, material);
            
            const group = new Group();
            group.add(outlineMesh);
            group.add(textMesh);
            
            // Center the group if bounding box exists
            if (textGeometry.boundingBox) {
                const width = textGeometry.boundingBox.max.x - textGeometry.boundingBox.min.x;
                group.position.x -= width / 2;
            }
            return group;
        }

        // Create mesh with the text geometry and center it
        const mesh = new Mesh(textGeometry as unknown as BufferGeometry, material);
        
        // Center the mesh if bounding box exists
        if (textGeometry.boundingBox) {
            const width = textGeometry.boundingBox.max.x - textGeometry.boundingBox.min.x;
            mesh.position.x -= width / 2;
        }
        return mesh;
    }

    private setGroupLayer(group: Object3D, enabled: boolean): void {
        if (enabled) {
            group.traverse(child => {
                child.layers.disable(0);
                child.layers.enable(1);
            });
            group.layers.disable(0);
            group.layers.enable(1);
        } else {
            group.traverse(child => {
                child.layers.enable(0);
                child.layers.enable(1);
            });
            group.layers.enable(0);
            group.layers.enable(1);
        }
    }

    public setXRMode(enabled: boolean): void {
        if (enabled) {
            this.labelGroup.traverse(child => {
                child.layers.disable(0);
                child.layers.enable(1);
            });
            this.labelGroup.layers.disable(0);
            this.labelGroup.layers.enable(1);
        } else {
            this.labelGroup.traverse(child => {
                child.layers.enable(0);
                child.layers.enable(1);
            });
            this.labelGroup.layers.enable(0);
            this.labelGroup.layers.enable(1);
        }
    }

    public dispose(): void {
        this.labelGroup.traverse(child => {
            if (child instanceof THREE.Mesh) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
            }
        });
        this.scene.remove(this.labelGroup);
    }
}
