import {
    Scene,
    PerspectiveCamera,
    InstancedMesh,
    Matrix4,
    Vector3,
    Mesh,
    Object3D,
    Quaternion,
    WebGLRenderer,
    BufferGeometry,
    Material,
    LineSegments
} from 'three';
import { Node } from '../core/types';
import { Settings } from '../types/settings';
import { MetadataVisualizer } from './MetadataVisualizer';
import { HologramManager } from './HologramManager';
import { XRHandWithHaptics } from '../types/xr';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { HologramShaderMaterial } from './HologramShaderMaterial';

export class EnhancedNodeManager {
    private scene: Scene;
    private settings: Settings;
    private nodes: Map<string, Mesh> = new Map();
    private nodeGeometry: BufferGeometry;
    private nodeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private dummy = new Object3D();
    private geometryFactory: GeometryFactory;
    private materialFactory: MaterialFactory;
    private isInstanced: boolean;
    private isHologram: boolean;
    private hologramMaterial: HologramShaderMaterial | null = null;
    private metadataMaterial: Material | null = null;

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        this.nodeGeometry = this.geometryFactory.getNodeGeometry(settings.visualization.nodes.quality);

        if (this.settings.visualization.nodes.enableHologram) {
            this.hologramMaterial = this.materialFactory.getHologramMaterial(settings);
        }

        this.metadataMaterial = this.materialFactory.getMetadataMaterial();

        this.setupInstancedMesh();
    }

    private setupInstancedMesh() {
        if (this.isInstanced) {
            this.instancedMesh = new InstancedMesh(this.nodeGeometry, this.nodeMaterial, 1000);
            this.instancedMesh.count = 0;
            this.scene.add(this.instancedMesh);
        }
    }

    public updateNodePositionsAndVelocities(nodes: { position: [number, number, number]; velocity: [number, number, number] }[]): void {
        if (!this.instancedMesh) return;
    
        nodes.forEach((node, i) => {
            this.dummy.position.set(node.position[0], node.position[1], node.position[2]);
    
            const velocityMagnitude = Math.sqrt(
                node.velocity[0] * node.velocity[0] +
                node.velocity[1] * node.velocity[1] +
                node.velocity[2] * node.velocity[2]
            );
            const scaleFactor = 1 + velocityMagnitude * 0.5;
            this.dummy.scale.set(scaleFactor, scaleFactor, scaleFactor);
    
            this.dummy.updateMatrix();
            if (this.instancedMesh) {
                this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
            }
        });
    
        this.instancedMesh!.instanceMatrix.needsUpdate = true;
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        const newGeometry = this.geometryFactory.getNodeGeometry(settings.visualization.nodes.quality);
        if (this.nodeGeometry !== newGeometry) {
            this.nodeGeometry = newGeometry;
            if (this.instancedMesh) {
                this.instancedMesh.geometry = this.nodeGeometry;
            }
        }

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

        if (settings.visualization.nodes.enableMetadataVisualization && this.metadataMaterial) {
            const serverSupportsMetadata = true;
            if (serverSupportsMetadata) {
                // Apply metadata material and update visualization
            } else {
                // Disable metadata visualization
            }
        }
    }

    private rebuildInstancedMesh() {
        if (this.isInstanced) {
            this.instancedMesh = new InstancedMesh(this.nodeGeometry, this.nodeMaterial, 1000);
            this.instancedMesh.count = 0;
            this.scene.add(this.instancedMesh);
        }
    }

    updateNodes(nodes: Node[]) {
        this.instancedMesh!.count = nodes.length;

        nodes.forEach((node, index) => {
            const metadata = {
                id: node.id,
                name: node.data.metadata?.name || '',
                commitAge: this.calculateCommitAge(node.data.metadata?.lastModified || Date.now()),
                hyperlinkCount: node.data.metadata?.links?.length || 0,
                importance: this.calculateImportance(node),
                position: {
                    x: node.data.position.x,
                    y: node.data.position.y,
                    z: node.data.position.z
                }
            };

            const matrix = new Matrix4();

            if (this.settings.visualization.nodes.enableMetadataShape) {
                const nodeMesh = this.metadataVisualizer.createNodeMesh(metadata);
                nodeMesh.position.set(metadata.position.x, metadata.position.y, metadata.position.z);
                this.scene.add(nodeMesh);
            } else {
                const scale = this.calculateNodeScale(metadata.importance);
                const position = new Vector3(metadata.position.x, metadata.position.y, metadata.position.z);
                matrix.compose(position, this.quaternion, new Vector3(scale, scale, scale));
                this.instancedMesh!.setMatrixAt(index, matrix);
            }

            this.nodes.set(node.id, this.instancedMesh!.children[index] as Mesh);
        });

        this.instancedMesh!.instanceMatrix.needsUpdate = true;
    }

    private calculateCommitAge(timestamp: number): number {
        const now = Date.now();
        return (now - timestamp) / (1000 * 60 * 60 * 24); // Convert to days
    }

    private calculateImportance(node: Node): number {
        const linkFactor = node.data.metadata?.links ? node.data.metadata.links.length / 20 : 0;
        const referenceFactor = node.data.metadata?.references ? node.data.metadata.references.length / 10 : 0;
        return Math.min(linkFactor + referenceFactor, 1);
    }

    private calculateNodeScale(importance: number): number {
        const [min, max] = this.settings.visualization.nodes.sizeRange;
        return min + (max - min) * importance;
    }

    update(deltaTime: number) {
        if (this.isHologram && this.hologramMaterial) {
            this.hologramMaterial.update(deltaTime);
        }

        if (this.settings.visualization.animations.enableNodeAnimations) {
            this.instancedMesh!.instanceMatrix.needsUpdate = true;
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
        this.instancedMesh!.geometry.dispose();
        this.instancedMesh!.material.dispose();
        if (this.isHologram && this.hologramMaterial) {
            this.hologramMaterial.dispose();
        }
        if (this.metadataMaterial) {
            this.metadataMaterial.dispose();
        }
        this.scene.remove(this.instancedMesh!);
    }

    public createNode(id: string, data: NodeData, metadata: any): void {
        // ... existing code ...

        // Remove this line:
        // const nodeMesh = this.metadataVisualizer.createNodeMesh(metadata);
    }
}
