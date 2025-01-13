import {
    Scene,
    InstancedMesh,
    Matrix4,
    Vector3,
    Quaternion,
    BufferGeometry,
    Material
} from 'three';
import { Settings } from '../types/settings';
import { Edge } from '../core/types';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';

export class EdgeManager {
    private scene: Scene;
    private edgeGeometry: BufferGeometry;
    private edgeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private geometryFactory: GeometryFactory;
    private materialFactory: MaterialFactory;
    private quaternion = new Quaternion();

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.geometryFactory = GeometryFactory.getInstance();
        this.materialFactory = MaterialFactory.getInstance();
        this.edgeGeometry = this.geometryFactory.getEdgeGeometry();
        this.edgeMaterial = this.materialFactory.getEdgeMaterial(settings);
        this.setupInstancedMesh();
    }

    private setupInstancedMesh() {
        this.instancedMesh = new InstancedMesh(this.edgeGeometry, this.edgeMaterial, 1000);
        this.instancedMesh.count = 0;
        this.scene.add(this.instancedMesh);
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.materialFactory.updateMaterial('edge', settings);
    }

    updateEdges(edges: Edge[]) {
        const mesh = this.instancedMesh;
        if (!mesh) return;
        
        mesh.count = edges.length;

        edges.forEach((edge, index) => {
            if (!edge.sourcePosition || !edge.targetPosition) return;
            
            const startPos = new Vector3(
                edge.sourcePosition.x,
                edge.sourcePosition.y,
                edge.sourcePosition.z
            );
            const endPos = new Vector3(
                edge.targetPosition.x,
                edge.targetPosition.y,
                edge.targetPosition.z
            );
            
            // Calculate edge direction and length
            const direction = endPos.clone().sub(startPos);
            const length = direction.length();
            
            // Position the edge at the midpoint between source and target
            const position = startPos.clone().add(direction.multiplyScalar(0.5));
            
            // Calculate rotation
            const matrix = new Matrix4();
            const scale = new Vector3(1, length, 1);
            matrix.compose(position, this.quaternion, scale);

            mesh.setMatrixAt(index, matrix);
        });

        mesh.instanceMatrix.needsUpdate = true;
    }

    dispose() {
        if (this.instancedMesh) {
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
            this.scene.remove(this.instancedMesh);
        }
    }
}
