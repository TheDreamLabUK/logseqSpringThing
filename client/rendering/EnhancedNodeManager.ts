import {
    Scene,
    PerspectiveCamera,
    InstancedMesh,
    Matrix4,
    Vector3,
    Mesh,
    Object3D,
    Quaternion,
    WebGLRenderer
} from 'three';
import { Node, Settings } from '../core/types';
import { MetadataVisualizer } from './MetadataVisualizer';
import { HologramManager } from './HologramManager';
import { XRHandWithHaptics } from '../xr/xrTypes';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';

export class EnhancedNodeManager {
    private readonly nodeInstances: InstancedMesh;
    private readonly metadataVisualizer: MetadataVisualizer;
    private readonly hologramManager: HologramManager;
    private readonly nodeDataMap = new Map<string, Matrix4>();
    private readonly quaternion = new Quaternion();
    private readonly camera: PerspectiveCamera;
    private readonly geometryFactory: GeometryFactory;
    private readonly materialFactory: MaterialFactory;

    constructor(
        private readonly scene: Scene,
        renderer: WebGLRenderer,
        private readonly settings: Settings
    ) {
        // Get the camera from the scene
        const camera = scene.children.find(child => child instanceof PerspectiveCamera) as PerspectiveCamera;
        if (!camera) {
            throw new Error('No PerspectiveCamera found in scene');
        }
        this.camera = camera;

        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();

        this.metadataVisualizer = new MetadataVisualizer(this.camera, this.scene, this.settings);
        this.hologramManager = new HologramManager(scene, renderer, settings);
        scene.add(this.hologramManager.getGroup());

        const geometry = this.geometryFactory.getNodeGeometry('high');
        const material = this.materialFactory.getNodeMaterial(settings);

        this.nodeInstances = new InstancedMesh(geometry, material, 1000);
        this.nodeInstances.count = 0;
        scene.add(this.nodeInstances);
    }

    handleSettingsUpdate(settings: Settings) {
        this.materialFactory.updateMaterial('node-basic', settings);
    }

    updateNodes(nodes: Node[]) {
        this.nodeInstances.count = nodes.length;

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

            if (this.settings.nodes.enableMetadataShape) {
                const nodeMesh = this.metadataVisualizer.createNodeMesh(metadata);
                nodeMesh.position.set(metadata.position.x, metadata.position.y, metadata.position.z);
                this.scene.add(nodeMesh);
            } else {
                const scale = this.calculateNodeScale(metadata.importance);
                const position = new Vector3(metadata.position.x, metadata.position.y, metadata.position.z);
                matrix.compose(position, this.quaternion, new Vector3(scale, scale, scale));
                this.nodeInstances.setMatrixAt(index, matrix);
            }

            this.nodeDataMap.set(node.id, matrix);
        });

        this.nodeInstances.instanceMatrix.needsUpdate = true;
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
        const [min, max] = this.settings.nodes.sizeRange;
        return min + (max - min) * importance;
    }

    update(deltaTime: number) {
        this.hologramManager.update(deltaTime);

        if (this.settings.animations.enableNodeAnimations) {
            this.nodeInstances.instanceMatrix.needsUpdate = true;
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
            this.hologramManager.handleInteraction(position);
        }
    }

    dispose() {
        this.nodeInstances.geometry.dispose();
        this.nodeInstances.material.dispose();
        this.metadataVisualizer.dispose();
        this.scene.remove(this.nodeInstances);
        this.scene.remove(this.hologramManager.getGroup());
    }
}
