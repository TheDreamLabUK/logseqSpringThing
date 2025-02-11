import {
    Scene,
    Vector3,
    Mesh,
    Object3D,
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
    private geometryFactory: GeometryFactory;
    private materialFactory: MaterialFactory;
    private isHologram: boolean = false;
    private hologramMaterial: HologramShaderMaterial | null = null;
    private metadataMaterial: Material | null = null;
    private metadataVisualizer: MetadataVisualizer;
    private camera: PerspectiveCamera;
    private updateFrameCount = 0;
    private readonly AR_UPDATE_FREQUENCY = 2;
    private readonly METADATA_DISTANCE_THRESHOLD = 50;
    private readonly ANIMATION_DISTANCE_THRESHOLD = 30;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;

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
        
        // Create geometry with the base size from settings
        const baseSize = this.settings.visualization.nodes.sizeRange[0];
        this.nodeGeometry = this.geometryFactory.getNodeGeometry(
            settings.visualization.nodes.quality,
            platformManager.isXRMode ? 'ar' : 'desktop',
            baseSize
        );
        
        this.nodeMaterial = this.materialFactory.getNodeMaterial(settings);
        this.isHologram = settings.visualization.nodes.enableHologram;

        if (this.settings.visualization.nodes.enableHologram) {
            this.hologramMaterial = this.materialFactory.getHologramMaterial(settings);
        }

        this.metadataMaterial = this.materialFactory.getMetadataMaterial();
        this.metadataVisualizer = new MetadataVisualizer(this.camera, this.scene, settings);
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        
        // Update materials
        this.nodeMaterial = this.materialFactory.getNodeMaterial(settings);
        this.nodes.forEach(node => {
            node.material = this.nodeMaterial;
        });

        // Update geometry if needed
        const baseSize = settings.visualization.nodes.sizeRange[0];
        const newGeometry = this.geometryFactory.getNodeGeometry(
            settings.visualization.nodes.quality,
            platformManager.isXRMode ? 'ar' : 'desktop',
            baseSize
        );
        
        if (this.nodeGeometry !== newGeometry) {
            this.nodeGeometry = newGeometry;
            this.nodes.forEach(node => {
                node.geometry = this.nodeGeometry;
            });
        }

        // Update material settings
        this.materialFactory.updateMaterial('node-basic', settings);
        this.materialFactory.updateMaterial('node-phong', settings);
        this.materialFactory.updateMaterial('edge', settings);
        if (this.isHologram) {
            this.materialFactory.updateMaterial('hologram', settings);
        }

        const newIsHologram = settings.visualization.nodes.enableHologram;
        this.isHologram = newIsHologram;

        // Handle metadata visualization
        if (settings.visualization.nodes.enableMetadataVisualization) {
            const cameraPosition = this.camera.position;
            const shouldShowMetadata = (position: Vector3) => {
                return position.distanceTo(cameraPosition) < this.METADATA_DISTANCE_THRESHOLD;
            };

            this.nodes.forEach((node) => {
                // Remove existing metadata
                node.children.slice().forEach((child: Object3D) => {
                    if (child instanceof Object3D && (child.userData as any).isMetadata) {
                        node.remove(child);
                    }
                });

                // Add new metadata visualization
                const metadata = node.userData as NodeMetadata;
                if (metadata && shouldShowMetadata(node.position)) {
                    this.metadataVisualizer.createMetadataLabel(metadata).then((group) => {
                        if (shouldShowMetadata(node.position)) {
                            node.add(group);
                        }
                    });
                }
            });
        } else {
            // Remove all metadata visualizations
            this.nodes.forEach(node => {
                node.children.slice().forEach((child: Object3D) => {
                    if (child instanceof Object3D && (child.userData as any).isMetadata) {
                        node.remove(child);
                    }
                });
            });
        }
    }

    updateNodes(nodes: { id: string, data: NodeData }[]) {
        nodes.forEach((node) => {
            const existingNode = this.nodes.get(node.id);
            if (existingNode) {
                existingNode.position.set(
                    node.data.position.x,
                    node.data.position.y,
                    node.data.position.z
                );
                return;
            }

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
            } else {
                const position = new Vector3(metadata.position.x, metadata.position.y, metadata.position.z);
                nodeMesh = new Mesh(this.nodeGeometry, this.nodeMaterial);
                nodeMesh.position.copy(position);
                nodeMesh.layers.enable(0);
                nodeMesh.layers.enable(1);
                nodeMesh.userData = metadata;

                if (this.settings.visualization.nodes.enableMetadataVisualization && 
                    position.distanceTo(this.camera.position) < this.METADATA_DISTANCE_THRESHOLD) {
                    this.metadataVisualizer.createMetadataLabel(metadata).then((group) => {
                        if (position.distanceTo(this.camera.position) < this.METADATA_DISTANCE_THRESHOLD) {
                            nodeMesh.add(group);
                        }
                    });
                }
            }

            this.scene.add(nodeMesh);
            this.nodes.set(node.id, nodeMesh);
        });
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

    update(deltaTime: number) {
        this.updateFrameCount++;
        const isARMode = platformManager.isXRMode;

        if (isARMode && this.updateFrameCount % this.AR_UPDATE_FREQUENCY !== 0) {
            return;
        }

        if (this.isHologram && this.hologramMaterial) {
            this.hologramMaterial.update(deltaTime);
        }

        if (this.settings.visualization.animations.enableNodeAnimations) {
            const cameraPosition = this.camera.position;
            this.scene.traverse((child: Object3D) => {
                if (child instanceof Mesh && 
                    (child.material as any).type === 'LineBasicMaterial' && 
                    child.position.distanceTo(cameraPosition) < this.ANIMATION_DISTANCE_THRESHOLD) {
                    child.rotateY(0.001 * deltaTime);
                }
            });
        }
    }

    handleHandInteraction(hand: XRHandWithHaptics) {
        const indexTip = hand.hand.joints['index-finger-tip'] as Object3D | undefined;
        if (indexTip) {
            if (this.isHologram && this.hologramMaterial && indexTip.matrixWorld) {
                const position = new Vector3().setFromMatrixPosition(indexTip.matrixWorld);
                this.hologramMaterial.handleInteraction(position);
            }
        }
    }

    dispose() {
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
        nodes.forEach((node) => {
            const existingNode = this.nodes.get(node.id);
            if (existingNode) {
                existingNode.position.set(
                    node.data.position[0],
                    node.data.position[1],
                    node.data.position[2]
                );
            }
        });
    }

    public setXRMode(enabled: boolean): void {
        this.nodes.forEach(node => {
            node.layers.set(enabled ? 1 : 0);
            node.traverse((child: Object3D) => {
                child.layers.set(enabled ? 1 : 0);
            });
        });
    }

    public getNodes(): Map<string, Mesh> {
        return this.nodes;
    }
}
