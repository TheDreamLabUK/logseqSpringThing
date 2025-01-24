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
import { Settings } from '../types/settings';
import { MetadataVisualizer } from './MetadataVisualizer';
import { XRHandWithHaptics } from '../types/xr';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { HologramShaderMaterial } from './materials/HologramShaderMaterial';

interface NodePosition {
    position: [number, number, number];
    velocity: [number, number, number];
}

export class EnhancedNodeManager {
    private scene: Scene;
    private settings: Settings;
    private nodes: Map<string, Mesh> = new Map();
    private nodeGeometry: BufferGeometry;
    private nodeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private tempMatrix = new Matrix4();
    private tempVector = new Vector3();
    private tempQuaternion = new Quaternion();

    public updateNodePositions(nodeData: NodePosition[]): void {
        const mesh = this.instancedMesh;
        if (!mesh) return;

        const scale = new Vector3(1, 1, 1);
        
        nodeData.forEach((data, index) => {
            const [x, y, z] = data.position;
            this.tempVector.set(x, y, z);
            this.tempMatrix.compose(
                this.tempVector,
                this.tempQuaternion,
                scale
            );
            mesh.setMatrixAt(index, this.tempMatrix);
        });

        // Mark instance matrix as needing update
        mesh.instanceMatrix.needsUpdate = true;
    }
    private dummy = new Object3D();
    private geometryFactory: GeometryFactory;
    private materialFactory: MaterialFactory;
    private isInstanced: boolean = false;
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
        this.nodeGeometry = this.geometryFactory.getNodeGeometry(settings.visualization.nodes.quality);
        this.nodeMaterial = this.materialFactory.getNodeMaterial(settings);

        if (this.settings.visualization.nodes.enableHologram) {
            this.hologramMaterial = this.materialFactory.getHologramMaterial(settings);
        }

        this.metadataMaterial = this.materialFactory.getMetadataMaterial();

        // Initialize MetadataVisualizer with both camera and scene
        this.metadataVisualizer = new MetadataVisualizer(this.camera, this.scene, settings);
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
    
            // Get base scale from settings
            const baseScale = this.settings.visualization.nodes.baseSize;
            
            // Calculate velocity-based scale with much smaller effect
            const velocityMagnitude = Math.sqrt(
                node.velocity[0] * node.velocity[0] +
                node.velocity[1] * node.velocity[1] +
                node.velocity[2] * node.velocity[2]
            );
            // Limit velocity effect to 10% increase
            const scaleFactor = baseScale * (1 + Math.min(velocityMagnitude * 0.1, 0.1));
            this.dummy.scale.set(scaleFactor, scaleFactor, scaleFactor);
    
            this.dummy.updateMatrix();
            this.instancedMesh?.setMatrixAt(i, this.dummy.matrix);
        });
    
        if (this.instancedMesh) {
            this.instancedMesh.instanceMatrix.needsUpdate = true;
        }
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

    updateNodes(nodes: { id: string, data: NodeData }[]) {
        const mesh = this.instancedMesh;
        if (!mesh) return;
        
        mesh.count = nodes.length;

        nodes.forEach((node, index) => {
            const metadata = {
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

            const matrix = new Matrix4();

            if (this.settings.visualization.nodes.enableMetadataShape) {
                const nodeMesh = this.metadataVisualizer.createNodeMesh(metadata);
                nodeMesh.position.set(metadata.position.x, metadata.position.y, metadata.position.z);
                this.scene.add(nodeMesh);
            } else {
                const scale = this.calculateNodeScale(metadata.importance);
                const position = new Vector3(metadata.position.x, metadata.position.y, metadata.position.z);
                matrix.compose(position, this.quaternion, new Vector3(scale, scale, scale));
                mesh.setMatrixAt(index, matrix);
            }

            const child = mesh.children[index];
            if (child instanceof Mesh) {
                this.nodes.set(node.id, child);
            }
        });

        mesh.instanceMatrix.needsUpdate = true;
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
    }

    public createNode(id: string, data: NodeData): void {
        const position = new Vector3(data.position.x, data.position.y, data.position.z);
        const scale = this.calculateNodeScale(this.calculateImportance({ id, data }));
        
        if (this.settings.visualization.nodes.enableMetadataShape) {
            const nodeMesh = this.metadataVisualizer.createNodeMesh({
                id,
                name: data.metadata?.name || '',
                commitAge: this.calculateCommitAge(data.metadata?.lastModified || Date.now()),
                hyperlinkCount: data.metadata?.links?.length || 0,
                importance: this.calculateImportance({ id, data }),
                position: data.position
            });
            this.nodes.set(id, nodeMesh);
            this.scene.add(nodeMesh);
        } else if (this.instancedMesh) {
            const matrix = new Matrix4();
            matrix.compose(position, this.quaternion, new Vector3(scale, scale, scale));
            const index = this.nodes.size;
            this.instancedMesh.setMatrixAt(index, matrix);
            this.instancedMesh.count = index + 1;
            this.instancedMesh.instanceMatrix.needsUpdate = true;
            // Store a reference to the instanced mesh for this node
            this.nodes.set(id, this.instancedMesh);
        }
    }
}
