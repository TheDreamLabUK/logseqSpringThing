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

// Reusable objects for matrix calculations to avoid garbage collection
const tempPosition = new Vector3();

// Batch processing constants
const MATRIX_UPDATE_BATCH_SIZE = 200;  // Increased to handle larger updates
const DISTANCE_UPDATE_THRESHOLD = 0.0001;  // Minimum distance for position updates
const MATRIX_UPDATE_INTERVAL = 2;  // Update matrices every N frames
const VIEW_CULLING_DISTANCE = 100;  // Maximum distance for node visibility

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
    private pendingMatrixUpdates: Set<string> = new Set();
    private matrixUpdateScheduled: boolean = false;
    private readonly METADATA_DISTANCE_THRESHOLD = 50;
    private readonly ANIMATION_DISTANCE_THRESHOLD = 30;
    private frameCount: number = 0;
    private visibleNodes: Set<string> = new Set();

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

    private updateVisibleNodes(): void {
        const cameraPosition = this.camera.position;
        this.visibleNodes.clear();

        this.nodes.forEach((node, id) => {
            const distance = node.position.distanceTo(cameraPosition);
            if (distance < VIEW_CULLING_DISTANCE) {
                this.visibleNodes.add(id);
            }
        });
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
                    Array.isArray(node.data.position) ? node.data.position[0] : node.data.position.x,
                    Array.isArray(node.data.position) ? node.data.position[1] : node.data.position.y,
                    Array.isArray(node.data.position) ? node.data.position[2] : node.data.position.z
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
                    x: Array.isArray(node.data.position) ? node.data.position[0] : node.data.position.x,
                    y: Array.isArray(node.data.position) ? node.data.position[1] : node.data.position.y,
                    z: Array.isArray(node.data.position) ? node.data.position[2] : node.data.position.z
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
        this.frameCount++;

        if (isARMode && this.updateFrameCount % this.AR_UPDATE_FREQUENCY !== 0) {
            return;
        }

        // Update visible nodes every N frames
        if (this.frameCount % MATRIX_UPDATE_INTERVAL === 0) {
            this.updateVisibleNodes();
        }

        if (this.isHologram && this.hologramMaterial) {
            this.hologramMaterial.update(deltaTime);
        }

        // Only process visible nodes
        for (const nodeId of this.visibleNodes) {
            const node = this.nodes.get(nodeId);
            if (!node) continue;

            const cameraPosition = this.camera.position;
            const distance = node.position.distanceTo(cameraPosition);

            // Only animate close nodes
            if (this.settings.visualization.animations.enableNodeAnimations &&
                distance < this.ANIMATION_DISTANCE_THRESHOLD) {
                node.rotateY(0.001 * deltaTime);
            }

            // Update matrices less frequently for better performance
            if (this.frameCount % MATRIX_UPDATE_INTERVAL === 0) {
                if (distance < this.METADATA_DISTANCE_THRESHOLD) {
                    node.updateMatrix();
                }
            }
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

    private scheduleBatchUpdate(): void {
        if (this.matrixUpdateScheduled) return;
        this.matrixUpdateScheduled = true;
        
        requestAnimationFrame(() => {
            this.processBatchUpdate();
            this.matrixUpdateScheduled = false;
            
            // If there are remaining updates, schedule another batch
            if (this.pendingMatrixUpdates.size > 0) {
                this.scheduleBatchUpdate();
            }
        });
    }

    private processBatchUpdate(): void {
        if (this.pendingMatrixUpdates.size === 0) return;

        let processed = 0;
        for (const nodeId of this.pendingMatrixUpdates) {
            if (processed >= MATRIX_UPDATE_BATCH_SIZE && this.pendingMatrixUpdates.size > MATRIX_UPDATE_BATCH_SIZE) {
                break; // Process remaining updates in next batch
            }

            const node = this.nodes.get(nodeId);
            if (node) {
                node.updateMatrixWorld(true);
                processed++;
                this.pendingMatrixUpdates.delete(nodeId);
            }
        }
    }

    public updateNodePositions(nodes: { id: string, data: { position: [number, number, number], velocity: [number, number, number] } }[]): void {
        nodes.forEach((node) => {
            const existingNode = this.nodes.get(node.id);
            if (!existingNode) return;

            // Skip tiny movements for performance
            const dx = node.data.position[0] - existingNode.position.x;
            const dy = node.data.position[1] - existingNode.position.y;
            const dz = node.data.position[2] - existingNode.position.z;
            const distanceSquared = dx * dx + dy * dy + dz * dz;
            
            if (distanceSquared < DISTANCE_UPDATE_THRESHOLD * DISTANCE_UPDATE_THRESHOLD) {
                return;
            }

            // Use reusable tempPosition Vector3
            tempPosition.set(
                node.data.position[0],
                node.data.position[1],
                node.data.position[2]
            );

            // Update position
            existingNode.position.copy(tempPosition);
            existingNode.updateMatrix();
            this.pendingMatrixUpdates.add(node.id);
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
