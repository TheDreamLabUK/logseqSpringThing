import {
    Scene,
    InstancedMesh,
    Matrix4,
    Vector3,
    Mesh,
    Object3D,
    Quaternion,
    BufferGeometry,
    Material,
    PerspectiveCamera
} from 'three';
import { NodeData } from '../core/types';
import { NodeMetadata } from '../types/metadata';
import { Settings } from '../types/settings/base';
import { MetadataVisualizer } from '../visualization/MetadataVisualizer';
import { XRHandWithHaptics } from '../types/xr';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { HologramShaderMaterial } from './materials/HologramShaderMaterial';
import { platformManager } from '../platform/platformManager';

export class EnhancedNodeManager {
    private scene: Scene;
    private settings: Settings;
    private nodes: Map<string, Mesh> = new Map();
    private nodeGeometry: BufferGeometry;
    private nodeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private geometryFactory: GeometryFactory;
    private materialFactory: MaterialFactory;
    private isInstanced: boolean;
    private isHologram: boolean = false;
    private hologramMaterial: HologramShaderMaterial | null = null;
    private metadataMaterial: Material | null = null;
    private metadataVisualizer: MetadataVisualizer;
    private quaternion = new Quaternion();
    private camera: PerspectiveCamera;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;

        // Find the camera in the scene
        let camera: PerspectiveCamera | null = null;
        scene.traverse((object) => {
            if (object instanceof PerspectiveCamera) {
                camera = object;
            }
        });
        if (!camera) {
            camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 5, 20);
            camera.lookAt(0, 0, 0);
            scene.add(camera);
        }
        this.camera = camera;

        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        this.isInstanced = settings.visualization.nodes.enableInstancing;
        this.nodeGeometry = this.geometryFactory.getNodeGeometry(settings.visualization.nodes.quality);
        this.nodeMaterial = this.materialFactory.getNodeMaterial(settings);
        this.isHologram = settings.visualization.nodes.enableHologram;

        if (this.settings.visualization.nodes.enableHologram) {
            this.hologramMaterial = this.materialFactory.getHologramMaterial(settings);
        }

        this.metadataMaterial = this.materialFactory.getMetadataMaterial();

        // Initialize MetadataVisualizer with camera and scene
        this.metadataVisualizer = new MetadataVisualizer(this.camera, this.scene, settings);
        this.setupInstancedMesh();
    }

    private setupInstancedMesh() {
        if (this.isInstanced) {
            this.instancedMesh = new InstancedMesh(this.nodeGeometry, this.nodeMaterial, 1000);
            this.instancedMesh.count = 0;
            this.instancedMesh.layers.set(platformManager.isXRMode ? 1 : 0);
            this.scene.add(this.instancedMesh);
        }
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        
        // Update materials
        this.nodeMaterial = this.materialFactory.getNodeMaterial(settings);
        if (!this.isInstanced) {
            this.nodes.forEach(node => {
                node.material = this.nodeMaterial;
            });
        }

        // Update geometry if needed
        const newGeometry = this.geometryFactory.getNodeGeometry(settings.visualization.nodes.quality);
        if (this.nodeGeometry !== newGeometry) {
            this.nodeGeometry = newGeometry;
            if (this.instancedMesh) {
                this.instancedMesh.geometry = this.nodeGeometry;
            }
            if (!this.isInstanced) {
                this.nodes.forEach(node => {
                    node.geometry = this.nodeGeometry;
                });
            }
        }

        // Update material settings
        this.materialFactory.updateMaterial('node-basic', settings);
        this.materialFactory.updateMaterial('node-phong', settings);
        this.materialFactory.updateMaterial('edge', settings);
        if (this.isHologram) {
            this.materialFactory.updateMaterial('hologram', settings);
        }

        const newIsInstanced = settings.visualization.nodes.enableInstancing;
        const newIsHologram = settings.visualization.nodes.enableHologram;

        if (newIsInstanced !== this.isInstanced || newIsHologram !== this.isHologram) {
            this.isInstanced = newIsInstanced;
            this.isHologram = newIsHologram;
            this.rebuildInstancedMesh();
        }

        // Handle metadata visualization
        if (settings.visualization.nodes.enableMetadataVisualization) {
            this.nodes.forEach((node) => {
                // Remove existing metadata
                node.children.slice().forEach(child => {
                    if (child instanceof Object3D && child.userData.isMetadata) {
                        node.remove(child);
                    }
                });

                // Add new metadata visualization
                const metadata = node.userData as NodeMetadata;
                if (metadata) {
                    this.metadataVisualizer.createMetadataLabel(metadata).then((group) => {
                        node.add(group);
                    });
                }
            });
        } else {
            // Remove all metadata visualizations
            this.nodes.forEach(node => {
                node.children.slice().forEach(child => {
                    if (child instanceof Object3D && child.userData.isMetadata) {
                        node.remove(child);
                    }
                });
            });
        }
    }

    updateNodes(nodes: { id: string, data: NodeData }[]) {
        // Clear existing nodes
        if (!this.isInstanced) {
            this.nodes.forEach(node => {
                this.scene.remove(node);
                if (node.geometry) node.geometry.dispose();
                if (node.material) node.material.dispose();
            });
            this.nodes.clear();
        }

        if (this.isInstanced && this.instancedMesh) {
            this.instancedMesh.count = nodes.length;
        }

        nodes.forEach((node, index) => {
            const metadata: NodeMetadata = {
                id: node.id,
                name: node.data.metadata?.name || '',
                commitAge: this.calculateCommitAge(node.data.metadata?.lastModified || Date.now()),
                hyperlinkCount: node.data.metadata?.links?.length || 0,
                importance: this.calculateImportance({ id: node.id, data: node.data }),
                position: {
                    x: node.data.position.x,
                    y: node.data.position.y,
                    z: node.data.position.z
                }
            };

            let nodeMesh: Mesh;

            if (this.settings.visualization.nodes.enableMetadataShape) {
                nodeMesh = this.metadataVisualizer.createNodeVisual(metadata);
                nodeMesh.position.set(metadata.position.x, metadata.position.y, metadata.position.z);
                nodeMesh.userData = metadata;

                if (this.settings.visualization.nodes.enableMetadataVisualization) {
                    this.metadataVisualizer.createMetadataLabel(metadata).then((group) => {
                        nodeMesh.add(group);
                    });
                }

                this.scene.add(nodeMesh);
                this.nodes.set(node.id, nodeMesh);
            } else {
                const scale = this.calculateNodeScale(metadata.importance);
                const position = new Vector3(metadata.position.x, metadata.position.y, metadata.position.z);

                if (this.isInstanced && this.instancedMesh) {
                    const matrix = new Matrix4();
                    matrix.compose(position, this.quaternion, new Vector3(scale, scale, scale));
                    this.instancedMesh.setMatrixAt(index, matrix);
                } else {
                    nodeMesh = new Mesh(this.nodeGeometry, this.nodeMaterial);
                    nodeMesh.position.copy(position);
                    nodeMesh.scale.set(scale, scale, scale);
                    nodeMesh.layers.enable(0);
                    nodeMesh.layers.enable(1);
                    nodeMesh.userData = metadata;

                    if (this.settings.visualization.nodes.enableMetadataVisualization) {
                        this.metadataVisualizer.createMetadataLabel(metadata).then((group) => {
                            nodeMesh.add(group);
                        });
                    }

                    this.scene.add(nodeMesh);
                    this.nodes.set(node.id, nodeMesh);
                }
            }
        });

        if (this.isInstanced && this.instancedMesh) {
            this.instancedMesh.instanceMatrix.needsUpdate = true;
        }
    }

    private calculateCommitAge(timestamp: number): number {
        const now = Date.now();
        return (now - timestamp) / (1000 * 60 * 60 * 24); // Convert to days
    }

    private calculateImportance(node: { id: string, data: NodeData }): number {
        const linkFactor = node.data.metadata?.links ? node.data.metadata.links.length / 20 : 0;
        const referenceFactor = node.data.metadata?.references ? node.data.metadata.references.length / 10 : 0;
        return Math.min(linkFactor + referenceFactor, 1);
    }

    private calculateNodeScale(importance: number): number {
        const [min, max] = this.settings.visualization.nodes.sizeRange;
        return min + (max - min) * importance;
    }

    private rebuildInstancedMesh() {
        if (this.instancedMesh) {
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
            this.scene.remove(this.instancedMesh);
        }
        if (this.isInstanced) {
            this.instancedMesh = new InstancedMesh(this.nodeGeometry, this.nodeMaterial, 1000);
            this.instancedMesh.count = 0;
            this.instancedMesh.layers.set(platformManager.isXRMode ? 1 : 0);
            this.scene.add(this.instancedMesh);
        }
    }

    update(deltaTime: number) {
        if (this.isHologram && this.hologramMaterial) {
            this.hologramMaterial.update(deltaTime);
        }

        if (this.settings.visualization.animations.enableNodeAnimations) {
            if (this.instancedMesh) {
                this.instancedMesh.instanceMatrix.needsUpdate = true;
            }
            this.scene.traverse(child => {
                if (child instanceof Mesh) {
                    child.rotateY(0.001 * deltaTime);
                }
            });
        }
    }

    handleHandInteraction(hand: XRHandWithHaptics) {
        const position = new Vector3();
        const indexTip = hand.hand.joints['index-finger-tip'] as Object3D | undefined;
        if (indexTip) {
            position.setFromMatrixPosition(indexTip.matrixWorld);
            if (this.isHologram && this.hologramMaterial) {
                this.hologramMaterial.handleInteraction(position);
            }
        }
    }

    dispose() {
        if (this.instancedMesh) {
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
            this.scene.remove(this.instancedMesh);
        }
        if (this.isHologram && this.hologramMaterial) {
            this.hologramMaterial.dispose();
        }
        if (this.metadataMaterial) {
            this.metadataMaterial.dispose();
        }
        this.nodes.forEach(node => {
            if (node.geometry) node.geometry.dispose();
            if (node.material) node.material.dispose();
            this.scene.remove(node);
        });
        this.nodes.clear();
    }

    public updateNodePositions(nodes: { id: string, data: { position: [number, number, number], velocity: [number, number, number] } }[]): void {
        nodes.forEach((node, index) => {
            const existingNode = this.nodes.get(node.id);
            if (existingNode) {
                existingNode.position.set(
                    node.data.position[0],
                    node.data.position[1],
                    node.data.position[2]
                );
            } else if (this.isInstanced && this.instancedMesh) {
                const matrix = new Matrix4();
                matrix.compose(
                    new Vector3(
                        node.data.position[0],
                        node.data.position[1],
                        node.data.position[2]
                    ),
                    this.quaternion,
                    new Vector3(1, 1, 1)
                );
                this.instancedMesh.setMatrixAt(index, matrix);
            }
        });

        if (this.isInstanced && this.instancedMesh) {
            this.instancedMesh.instanceMatrix.needsUpdate = true;
        }
    }

    public setXRMode(enabled: boolean): void {
        if (this.instancedMesh) {
            this.instancedMesh.layers.set(enabled ? 1 : 0);
        }
        this.nodes.forEach(node => {
            node.layers.set(enabled ? 1 : 0);
            node.traverse(child => {
                child.layers.set(enabled ? 1 : 0);
            });
        });
    }
}
