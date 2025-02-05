import {
    Scene,
    InstancedMesh,
    Matrix4,
    Vector3,
    BufferGeometry,
    Object3D,
    Material
} from 'three';
import { Settings } from '../types/settings';
import { Edge } from '../core/types';
import { MaterialFactory } from './factories/MaterialFactory';
import { GeometryFactory } from './factories/GeometryFactory';

export class EdgeManager {
    private scene: Scene;
    private edgeGeometry: BufferGeometry;
    private edgeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private materialFactory: MaterialFactory;
    private settings: Settings;

    // Reusable objects for calculations
    private readonly startPos = new Vector3();
    private readonly endPos = new Vector3();
    private readonly tempObject = new Object3D();
    private readonly direction = new Vector3();
    private readonly position = new Vector3();
    private readonly matrix = new Matrix4();

    // Batch size for matrix updates
    private static readonly BATCH_SIZE = 1000;
    private pendingUpdates: Set<number> = new Set();
    private updateScheduled = false;

    constructor(scene: Scene, settings: Settings) {
        this.settings = settings;
        this.scene = scene;
        this.materialFactory = MaterialFactory.getInstance();
        
        // Get edge geometry from factory
        const geometryFactory = GeometryFactory.getInstance();
        this.edgeGeometry = geometryFactory.getEdgeGeometry('desktop');
        
        this.edgeMaterial = this.materialFactory.getEdgeMaterial(settings);
        this.setupInstancedMesh();
    }

    private setupInstancedMesh() {
        this.instancedMesh = new InstancedMesh(this.edgeGeometry, this.edgeMaterial, 1000);
        this.instancedMesh.count = 0;
        this.instancedMesh.frustumCulled = true; // Enable frustum culling
        this.scene.add(this.instancedMesh);
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        this.materialFactory.updateMaterial('edge', settings);
        
        // Update all edges with new settings
        if (this.currentEdges) {
            this.updateEdges(this.currentEdges);
        }
    }

    private scheduleBatchUpdate(): void {
        if (this.updateScheduled) return;
        this.updateScheduled = true;

        requestAnimationFrame(() => {
            this.processBatchUpdate();
            this.updateScheduled = false;
        });
    }

    private processBatchUpdate(): void {
        if (!this.instancedMesh || this.pendingUpdates.size === 0) return;

        let processed = 0;
        this.pendingUpdates.forEach(index => {
            if (processed >= EdgeManager.BATCH_SIZE) return;

            const currentEdge = this.currentEdges[index];
            if (!currentEdge?.sourcePosition || !currentEdge?.targetPosition) return;

            // Set positions
            this.startPos.set(
                currentEdge.sourcePosition.x,
                currentEdge.sourcePosition.y,
                currentEdge.sourcePosition.z
            );
            this.endPos.set(
                currentEdge.targetPosition.x,
                currentEdge.targetPosition.y,
                currentEdge.targetPosition.z
            );

            const edge = this.currentEdges[index];
            if (!edge?.sourcePosition || !edge?.targetPosition) return;

            // Set positions
            this.startPos.set(
                edge.sourcePosition.x,
                edge.sourcePosition.y,
                edge.sourcePosition.z
            );
            this.endPos.set(
                edge.targetPosition.x,
                edge.targetPosition.y,
                edge.targetPosition.z
            );

            // Calculate direction and length
            this.direction.subVectors(this.endPos, this.startPos);
            const length = this.direction.length();
            
            if (length === 0) return;

            // Position at midpoint
            this.position.addVectors(this.startPos, this.endPos).multiplyScalar(0.5);
            
            // Calculate direction
            this.direction.normalize();
            
            // Use Object3D to handle transformations
            this.tempObject.position.copy(this.position);

            // Scale width based on settings and edge length
            const baseWidth = this.settings.visualization?.edges?.baseWidth || 2.0;
            const widthRange = this.settings.visualization?.edges?.widthRange || [1.0, 3.0];
            const lengthFactor = Math.max(0.5, Math.min(1.5, length / 10)); // Scale width based on length
            const edgeWidth = baseWidth * 0.05 * lengthFactor; // Convert baseWidth to scene units
            const clampedWidth = Math.max(widthRange[0], Math.min(widthRange[1], edgeWidth));
            
            this.tempObject.scale.set(clampedWidth, length, clampedWidth);
            this.tempObject.lookAt(this.endPos);
            this.tempObject.updateMatrix();
            
            // Copy the transformation matrix
            this.matrix.copy(this.tempObject.matrix);
            
            this.instancedMesh!.setMatrixAt(index, this.matrix);
            
            processed++;
            this.pendingUpdates.delete(index);
        });

        if (processed > 0) {
            this.instancedMesh.instanceMatrix.needsUpdate = true;
        }

        if (this.pendingUpdates.size > 0) {
            this.scheduleBatchUpdate();
        }
    }

    private currentEdges: Edge[] = [];

    updateEdges(edges: Edge[]) {
        const mesh = this.instancedMesh;
        if (!mesh) return;
        
        mesh.count = edges.length;
        this.currentEdges = edges;

        // Queue all edges for update
        for (let i = 0; i < edges.length; i++) {
            this.pendingUpdates.add(i);
        }

        this.scheduleBatchUpdate();
    }

    dispose() {
        if (this.instancedMesh) {
            this.instancedMesh.geometry.dispose();
            this.instancedMesh.material.dispose();
            this.scene.remove(this.instancedMesh);
        }
        this.pendingUpdates.clear();
    }
}
