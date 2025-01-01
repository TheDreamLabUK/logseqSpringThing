import {
    Scene,
    InstancedMesh,
    Mesh,
    Object3D,
    Quaternion,
    BufferGeometry,
    Material,
    PerspectiveCamera
} from 'three';
import { NodeData, NodeMesh, VisualizationSettings } from '../core/types';
import { NodeMeshUserData } from '../types/settings';
import { MetadataVisualizer } from './MetadataVisualizer';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { HologramShaderMaterial } from './materials/HologramShaderMaterial';

export class EnhancedNodeManager {
    private scene: Scene;
    private settings: VisualizationSettings;
    private nodes: Map<string, NodeMesh> = new Map();
    private nodeGeometry: BufferGeometry;
    private nodeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private dummy = new Object3D();
    private geometryFactory: GeometryFactory;
    private materialFactory: MaterialFactory;
    private isInstanced: boolean = false;
    private isHologram: boolean = false;
    private hologramMaterial: HologramShaderMaterial | null = null;
    private metadataVisualizer: MetadataVisualizer;
    private quaternion = new Quaternion();
    private camera: PerspectiveCamera;

    constructor(scene: Scene, settings: VisualizationSettings, camera: PerspectiveCamera) {
        this.scene = scene;
        this.settings = settings;
        this.camera = camera;

        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        this.nodeGeometry = this.geometryFactory.getNodeGeometry('medium');
        
        const materialType = this.settings.nodes.material.type;
        this.nodeMaterial = this.materialFactory.createNodeMaterial(materialType);

        if (this.settings.hologram) {
            this.hologramMaterial = this.materialFactory.createHologramMaterial();
            this.isHologram = true;
        }

        // Initialize MetadataVisualizer
        this.metadataVisualizer = new MetadataVisualizer(this.camera, this.scene, settings);
        
        if (this.isInstanced) {
            this.setupInstancedMesh();
        }
    }

    private setupInstancedMesh() {
        if (this.isInstanced && this.nodeGeometry && this.nodeMaterial) {
            this.instancedMesh = new InstancedMesh(
                this.nodeGeometry,
                this.nodeMaterial,
                1000
            );
            this.instancedMesh.count = 0;
            this.scene.add(this.instancedMesh);
        }
    }

    public updateNodePositions(nodes: NodeData[]): void {
        if (this.isInstanced && this.instancedMesh) {
            nodes.forEach((node, i) => {
                this.dummy.position.copy(node.position);
                this.dummy.scale.setScalar(node.size || this.settings.nodes.defaultSize);
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
                    nodeMesh.scale.setScalar(node.size || this.settings.nodes.defaultSize);
                }
            });
        }
    }

    public addNode(node: NodeData): void {
        if (this.nodes.has(node.id)) {
            return;
        }

        const materialType = this.settings.nodes.material.type;
        const material = this.materialFactory.createNodeMaterial(materialType, node.color);
        
        const mesh = new Mesh(this.nodeGeometry, material) as NodeMesh;
        mesh.position.copy(node.position);
        mesh.scale.setScalar(node.size || this.settings.nodes.defaultSize);
        
        mesh.userData = {
            id: node.id,
            type: 'node',
            properties: node.properties,
            data: node
        } as NodeMeshUserData;

        this.nodes.set(node.id, mesh);
        this.scene.add(mesh);

        // Add metadata if enabled
        if (this.settings.nodes.material.transparent) {
            this.metadataVisualizer.addNodeMetadata(node);
        }
    }

    public removeNode(nodeId: string): void {
        const node = this.nodes.get(nodeId);
        if (node) {
            if (node.material instanceof Array) {
                node.material.forEach(m => m.dispose());
            } else {
                node.material.dispose();
            }
            this.scene.remove(node);
            this.nodes.delete(nodeId);
        }
    }

    public updateNodeMaterial(nodeId: string, material: Material): void {
        const node = this.nodes.get(nodeId);
        if (node) {
            if (node.material instanceof Array) {
                node.material.forEach(m => m.dispose());
            } else {
                node.material.dispose();
            }
            node.material = material;
        }
    }

    public updateSettings(settings: VisualizationSettings): void {
        this.settings = settings;
        this.materialFactory.updateMaterial('node', settings);
        
        if (this.isHologram && this.hologramMaterial) {
            this.materialFactory.updateMaterial('hologram', settings);
        }
    }

    public dispose(): void {
        this.nodes.forEach(node => {
            if (node.material instanceof Array) {
                node.material.forEach(m => m.dispose());
            } else {
                node.material.dispose();
            }
            this.scene.remove(node);
        });
        this.nodes.clear();

        if (this.instancedMesh) {
            this.instancedMesh.dispose();
            this.scene.remove(this.instancedMesh);
            this.instancedMesh = null;
        }

        this.nodeGeometry.dispose();
        this.metadataVisualizer.dispose();
    }
}
