import {
    Scene,
    InstancedMesh,
    Matrix4,
    Vector3,
    BufferGeometry,
    Object3D,
    Material,
    Quaternion
} from 'three';
import { Settings } from '../types/settings';
import { Edge } from '../core/types';
import { MaterialFactory } from './factories/MaterialFactory';
import { GeometryFactory } from './factories/GeometryFactory';
import { createLogger } from '../core/logger';

const logger = createLogger('EdgeManager');

export class EdgeManager {
    private scene: Scene;
    private edgeGeometry!: BufferGeometry;
    private edgeMaterial: Material;
    private instancedMesh: InstancedMesh | null = null;
    private materialFactory: MaterialFactory;
    private geometryFactory: GeometryFactory;
    private settings: Settings;

    // Quality settings
    private currentQuality: 'low' | 'medium' | 'high' = 'medium';
    private currentContext: 'ar' | 'desktop' = 'desktop';

    // Reusable objects for calculations
    private readonly startPos = new Vector3();
    private readonly endPos = new Vector3();
    private readonly tempObject = new Object3D();
    private readonly direction = new Vector3();
    private readonly position = new Vector3();
    private readonly matrix = new Matrix4();
    private readonly upVector = new Vector3(0, 1, 0);
    private readonly quaternion = new Quaternion();

    // Batch size for matrix updates
    private static readonly BATCH_SIZE = 1000;
    private pendingUpdates: Set<number> = new Set();
    private updateScheduled = false;

    constructor(scene: Scene, settings: Settings) {
        this.settings = settings;
        this.scene = scene;
        this.materialFactory = MaterialFactory.getInstance();
        this.geometryFactory = GeometryFactory.getInstance();
        
        // Initialize with settings
        this.updateGeometryFromSettings(settings);
        this.edgeMaterial = this.materialFactory.getEdgeMaterial(settings);
        this.setupInstancedMesh();
    }

    private updateGeometryFromSettings(settings: Settings): void {
        // Get quality and context from settings
        const quality = settings.visualization?.edges?.quality || 'medium';
        const context = settings.visualization?.rendering?.context || 'desktop';

        // Only update if quality or context has changed
        if (quality !== this.currentQuality || context !== this.currentContext) {
            this.currentQuality = quality;
            this.currentContext = context;
            
            // Get new geometry with updated quality
            this.edgeGeometry = this.geometryFactory.getEdgeGeometry(this.currentContext, this.currentQuality);
            
            // Update instanced mesh if it exists
            if (this.instancedMesh) {
                const oldMesh = this.instancedMesh;
                this.setupInstancedMesh();
                oldMesh.geometry.dispose();
                this.scene.remove(oldMesh);
                
                // Reapply current edges with new geometry
                if (this.currentEdges.length > 0) {
                    this.updateEdges(this.currentEdges);
                }
            }
        }
    }

    public handleSettingsUpdate(settings: Settings): void {
        const oldSettings = this.settings;
        this.settings = settings;
        
        // Check if we need to update geometry
        if (settings.visualization?.edges?.quality !== oldSettings.visualization?.edges?.quality ||
            settings.visualization?.rendering?.context !== oldSettings.visualization?.rendering?.context) {
            this.updateGeometryFromSettings(settings);
        }
        
        // Update material
        this.materialFactory.updateMaterial('edge', settings);
        
        // Update all edges with new settings
        if (this.currentEdges) {
            this.updateEdges(this.currentEdges);
        }
    }

    private setupInstancedMesh() {
        // Use a larger initial capacity for better performance
        const initialCapacity = Math.max(1000, this.currentEdges?.length || 0);
        this.instancedMesh = new InstancedMesh(this.edgeGeometry, this.edgeMaterial, initialCapacity);
        this.instancedMesh.count = 0;
        this.instancedMesh.frustumCulled = true; // Enable frustum culling
        this.scene.add(this.instancedMesh);
    }

    private validateEdgePositions(start: Vector3, end: Vector3): boolean {
        if (start.distanceTo(end) < 0.0001) {
            logger.warn('Edge endpoints too close', {
                start: `${start.x},${start.y},${start.z}`,
                end: `${end.x},${end.y},${end.z}`
            });
            return false;
        }
        return true;
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

            // Validate positions
            if (!this.validateEdgePositions(this.startPos, this.endPos)) return;

            // Calculate direction and length
            this.direction.subVectors(this.endPos, this.startPos);
            const length = this.direction.length();
            
            // Position at midpoint
            this.position.copy(this.startPos).add(this.direction.multiplyScalar(0.5));
            
            // Calculate rotation
            this.direction.normalize();
            const angle = Math.acos(this.direction.dot(this.upVector));
            const axis = new Vector3().crossVectors(this.upVector, this.direction).normalize();
            this.quaternion.setFromAxisAngle(axis, angle);
            
            // Use Object3D to handle transformations
            this.tempObject.position.copy(this.position);
            this.tempObject.quaternion.copy(this.quaternion);

            // Get edge width settings with proper defaults
            const edgeSettings = this.settings.visualization?.edges || {};
            const baseWidth = edgeSettings.baseWidth || 1.0;
            const widthRange = edgeSettings.widthRange || [0.5, 2.0];
            const scaleFactor = edgeSettings.scaleFactor || 1.0;
            
            // Calculate edge width based on settings
            const edgeWidth = baseWidth * scaleFactor;
            const clampedWidth = Math.max(widthRange[0], Math.min(widthRange[1], edgeWidth));
            
            // Apply scale with proper proportions
            // Note: The cylinder's base radius is already 1.0 from GeometryFactory
            this.tempObject.scale.set(clampedWidth, length, clampedWidth);
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
