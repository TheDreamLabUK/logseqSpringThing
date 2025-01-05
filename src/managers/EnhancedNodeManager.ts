import { Scene, Mesh, Material, Vector3, Matrix4, InstancedMesh, Object3D, SphereGeometry, MeshBasicMaterial, Color } from 'three';
import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';

export class EnhancedNodeManager {
    private logger = createLogger('EnhancedNodeManager');
    private scene: Scene;
    private settings: Settings;
    private nodes: Map<string, Mesh>;
    private isInstanced: boolean;
    private instancedMesh: InstancedMesh | null;
    private dummy: Object3D;

    constructor(scene: Scene, settings: Settings) {
        this.logger.debug('Initializing EnhancedNodeManager');
        this.scene = scene;
        this.settings = settings;
        this.nodes = new Map();
        this.isInstanced = settings.visualization.nodes.enableInstancing;
        this.instancedMesh = null;
        this.dummy = new Object3D();
    }

    initialize(nodes: Array<{ id: string, position: Vector3, size?: number }>) {
        this.logger.debug('Initializing nodes', { count: nodes.length });
        
        if (this.isInstanced) {
            this.initializeInstanced(nodes);
        } else {
            this.initializeIndividual(nodes);
        }
    }

    private initializeInstanced(nodes: Array<{ id: string, position: Vector3, size?: number }>) {
        const geometry = new SphereGeometry(1, 32, 32);
        const material = new MeshBasicMaterial({
            color: new Color(this.settings.visualization.nodes.color),
            transparent: true,
            opacity: this.settings.visualization.nodes.opacity
        });

        this.instancedMesh = new InstancedMesh(geometry, material, nodes.length);
        this.scene.add(this.instancedMesh);

        // Set initial positions
        nodes.forEach((node, i) => {
            this.dummy.position.copy(node.position);
            this.dummy.scale.setScalar(node.size || this.settings.visualization.nodes.defaultSize);
            this.dummy.updateMatrix();
            this.instancedMesh?.setMatrixAt(i, this.dummy.matrix);
        });

        if (this.instancedMesh) {
            this.instancedMesh.instanceMatrix.needsUpdate = true;
        }
    }

    private initializeIndividual(nodes: Array<{ id: string, position: Vector3, size?: number }>) {
        const geometry = new SphereGeometry(1, 32, 32);
        
        nodes.forEach(node => {
            const material = new MeshBasicMaterial({
                color: new Color(this.settings.visualization.nodes.color),
                transparent: true,
                opacity: this.settings.visualization.nodes.opacity
            });

            const mesh = new Mesh(geometry, material);
            mesh.position.copy(node.position);
            mesh.scale.setScalar(node.size || this.settings.visualization.nodes.defaultSize);
            
            this.nodes.set(node.id, mesh);
            this.scene.add(mesh);
        });
    }

    updateNodePositions(nodes: Array<{ id: string, position: Vector3, size?: number }>) {
        this.logger.debug('Updating node positions', { nodeCount: nodes.length });
        
        if (this.isInstanced && this.instancedMesh) {
            nodes.forEach((node, i) => {
                this.dummy.position.copy(node.position);
                this.dummy.scale.setScalar(node.size || this.settings.visualization.nodes.defaultSize);
                this.dummy.updateMatrix();
                this.instancedMesh?.setMatrixAt(i, this.dummy.matrix);
            });
            
            if (this.instancedMesh) {
                this.instancedMesh.instanceMatrix.needsUpdate = true;
            }
        } else {
            nodes.forEach(node => {
                const nodeMesh = this.nodes.get(node.id);
                if (nodeMesh) {
                    nodeMesh.position.copy(node.position);
                    nodeMesh.scale.setScalar(node.size || this.settings.visualization.nodes.defaultSize);
                }
            });
        }
    }

    updateNodeVisibility(nodeId: string, visible: boolean) {
        const node = this.nodes.get(nodeId);
        if (node) {
            node.visible = visible;
        }
    }

    dispose() {
        this.logger.debug('Disposing EnhancedNodeManager');
        
        if (this.isInstanced && this.instancedMesh) {
            this.scene.remove(this.instancedMesh);
            this.instancedMesh.geometry.dispose();
            (this.instancedMesh.material as Material).dispose();
            this.instancedMesh = null;
        } else {
            this.nodes.forEach(node => {
                this.scene.remove(node);
                node.geometry.dispose();
                (node.material as Material).dispose();
            });
            this.nodes.clear();
        }
    }

    // Add other necessary methods...
} 