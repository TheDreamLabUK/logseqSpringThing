import * as THREE from 'three';
import {
    Scene,
    PerspectiveCamera,
    InstancedMesh,
    SphereBufferGeometry,
    MeshBasicMaterial,
    Matrix4,
    Vector3,
    Mesh,
    Object3D,
    Quaternion
} from 'three';
import { Node, Settings } from '../core/types';
import { MetadataVisualizer } from './MetadataVisualizer';
import { HologramManager } from './HologramManager';
import { XRHandWithHaptics } from '../xr/xrTypes';

export class EnhancedNodeManager {
    private readonly nodeInstances: InstancedMesh;
    private readonly metadataVisualizer: MetadataVisualizer;
    private readonly hologramManager: HologramManager;
    private readonly nodeDataMap = new Map<string, Matrix4>();
    private readonly quaternion = new Quaternion();

    constructor(
        private readonly scene: Scene,
        private readonly camera: PerspectiveCamera,
        private readonly currentSettings: Settings
    ) {
        this.metadataVisualizer = new MetadataVisualizer(camera, currentSettings);
        this.hologramManager = new HologramManager(scene, camera, currentSettings);
        scene.add(this.hologramManager.getGroup());

        const geometry = new THREE.SphereGeometry(1, 32, 32);
        const material = new MeshBasicMaterial({
            color: this.currentSettings.nodes.baseColor,
            transparent: true,
            opacity: this.currentSettings.nodes.opacity
        });

        this.nodeInstances = new InstancedMesh(geometry, material, 1000);
        this.nodeInstances.count = 0;
        scene.add(this.nodeInstances);
    }

    handleSettingsUpdate(settings: Settings) {
        this.updateMaterials(settings);
    }

    updateNodes(nodes: Node[]) {
        this.nodeInstances.count = nodes.length;

        nodes.forEach((node, index) => {
            const metadata = {
                name: node.data.metadata?.name || '',
                commitAge: this.calculateCommitAge(node.data.metadata?.lastModified || Date.now()),
                hyperlinkCount: node.data.metadata?.links?.length || 0,
                importance: this.calculateImportance(node),
                position: new Vector3(
                    node.data.position.x,
                    node.data.position.y,
                    node.data.position.z
                )
            };

            const matrix = new Matrix4();

            if (this.currentSettings.nodes.enableMetadataShape) {
                const nodeMesh = this.metadataVisualizer.createNodeMesh(metadata);
                nodeMesh.position.copy(metadata.position);
                this.scene.add(nodeMesh);
            } else {
                const scale = this.calculateNodeScale(metadata.importance);
                matrix.compose(metadata.position, this.quaternion, new Vector3(scale, scale, scale));
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
        const [min, max] = this.currentSettings.nodes.sizeRange;
        return min + (max - min) * importance;
    }

    update(deltaTime: number) {
        this.hologramManager.update(deltaTime);

        if (this.currentSettings.animations.enableNodeAnimations) {
            this.nodeInstances.instanceMatrix.needsUpdate = true;
            this.scene.traverse(child => {
                if (child instanceof Mesh) {
                    child.rotateY(0.001 * deltaTime);
                }
            });
        }
    }

    private updateMaterials(settings: Settings) {
        const material = this.nodeInstances.material as MeshBasicMaterial;
        if (material && settings.nodes.baseColor) {
            material.color.set(settings.nodes.baseColor);
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
