import {
    Scene,
    InstancedMesh,
    Matrix4,
    Vector3,
    Quaternion,
    BufferGeometry,
    Material,
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
    private settings: Settings;
    private static UP = new Vector3(0, 1, 0);

    constructor(scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
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
        this.settings = settings;
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
            direction.normalize();
            
            // Position the edge at the midpoint between source and target
            const position = startPos.clone().add(direction.clone().multiplyScalar(length * 0.5));
            
            // Get edge width settings from settings
            const edgeSettings = this.settings.visualization?.edges || {};
            const baseWidth = edgeSettings.baseWidth || 1.0;
            const scaleFactor = edgeSettings.scaleFactor || 1.0;
            const widthRange = edgeSettings.widthRange || [0.5, 2.0];
            
            // Calculate edge width and clamp to range
            const edgeWidth = Math.max(widthRange[0], Math.min(widthRange[1], baseWidth));
            
            // Create scale with width and length
            const scale = new Vector3(edgeWidth * scaleFactor, length * scaleFactor, edgeWidth * scaleFactor);
            
            // Calculate rotation from UP vector to edge direction
            const upDot = EdgeManager.UP.dot(direction);
            const rotationAxis = new Vector3().crossVectors(EdgeManager.UP, direction).normalize();
            const rotationAngle = Math.acos(upDot);
            
            // Create matrix from components
            const matrix = new Matrix4().compose(
                position,
                Math.abs(Math.abs(upDot) - 1) > 0.001
                    ? new Quaternion().setFromAxisAngle(rotationAxis, rotationAngle)
                    : new Quaternion(),
                scale
            );

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
