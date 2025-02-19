import {
    Scene,
    InstancedMesh,
    Matrix4,
    Vector3,
    Quaternion,
    Color,
    Camera,
    Material
} from 'three';
import { NodeGeometryManager, LODLevel } from '../geometry/NodeGeometryManager';
import { createLogger } from '../../../core/logger';
import { SettingsStore } from '../../../state/SettingsStore';
import { NodeSettings } from '../../../types/settings/base';
import { scaleOps } from '../../../core/utils';
import { Node } from '../../../core/types';

const logger = createLogger('NodeInstanceManager');

// Constants for optimization
const MAX_INSTANCES = 10000;
const VISIBILITY_UPDATE_INTERVAL = 10; // frames
const DEFAULT_FILE_SIZE = 1000; // 1KB default
const MAX_FILE_SIZE = 10485760; // 10MB max for scaling

// Reusable objects for matrix calculations
const matrix = new Matrix4();
const position = new Vector3();
const quaternion = new Quaternion();
const velocity = new Vector3();
const scale = new Vector3();

// Visibility states (using setRGB for proper initialization)
const VISIBLE = new Color(0xffffff);
const INVISIBLE = new Color(0x000000);

interface NodeUpdate {
    id: string;
    position: [number, number, number];
    velocity?: [number, number, number];
    metadata?: {
        name?: string;
        lastModified?: number;
        links?: string[];
        references?: string[];
        fileSize?: number;
        hyperlinkCount?: number;
    };
}

export class NodeInstanceManager {
    private static instance: NodeInstanceManager;
    private scene: Scene;
    private nodeInstances: InstancedMesh;
    private geometryManager: NodeGeometryManager;
    private nodeIndices: Map<string, number> = new Map();
    private pendingUpdates: Set<number> = new Set();
    private frameCount: number = 0;
    private velocities: Map<number, Vector3> = new Map();
    private lastUpdateTime: number = performance.now();
    private settingsStore: SettingsStore;
    private nodeSettings: NodeSettings;
    private readonly MAX_POSITION = 1000.0; // Maximum allowed position value
    private readonly MAX_VELOCITY = 10.0;   // Maximum allowed velocity value

    private validateVector3(vec: Vector3, max: number): boolean {
        return !isNaN(vec.x) && !isNaN(vec.y) && !isNaN(vec.z) &&
               isFinite(vec.x) && isFinite(vec.y) && isFinite(vec.z) &&
               Math.abs(vec.x) <= max && Math.abs(vec.y) <= max && Math.abs(vec.z) <= max;
    }

    private clampVector3(vec: Vector3, max: number): void {
        vec.x = Math.max(Math.min(vec.x, max), -max);
        vec.y = Math.max(Math.min(vec.y, max), -max);
        vec.z = Math.max(Math.min(vec.z, max), -max);
    }

    private constructor(scene: Scene, material: Material) {
        this.scene = scene;
        this.geometryManager = NodeGeometryManager.getInstance();
        this.settingsStore = SettingsStore.getInstance();
        this.nodeSettings = this.settingsStore.get('visualization.nodes') as NodeSettings;

        // Initialize InstancedMesh with high-detail geometry
        const initialGeometry = this.geometryManager.getGeometryForDistance(0);
        this.nodeInstances = new InstancedMesh(initialGeometry, material, MAX_INSTANCES);
        this.nodeInstances.count = 0; // Start with 0 visible instances
        this.nodeInstances.frustumCulled = true;
        this.nodeInstances.layers.enable(0); // Enable default layer

        // Add to scene
        this.scene.add(this.nodeInstances);
        logger.info('Initialized NodeInstanceManager');

        // Subscribe to settings changes
        this.settingsStore.subscribe('visualization.nodes', (_: string, settings: any) => {
            if (settings && typeof settings === 'object') {
                this.nodeSettings = settings as NodeSettings;
                this.updateAllNodeScales();
            }
        });
    }

    public static getInstance(scene: Scene, material: Material): NodeInstanceManager {
        if (!NodeInstanceManager.instance) {
            NodeInstanceManager.instance = new NodeInstanceManager(scene, material);
        }
        return NodeInstanceManager.instance;
    }

    private getNodeScale(node: Node): number {
        const [minSize, maxSize] = this.nodeSettings.sizeRange;
        
        // Get file size from metadata, use default if not available
        const fileSize = node.data.metadata?.fileSize || DEFAULT_FILE_SIZE;
        
        // Map file size logarithmically to 0-1 range
        // This gives better visual distribution since file sizes vary greatly
        const normalizedSize = Math.log(Math.min(fileSize, MAX_FILE_SIZE)) / Math.log(MAX_FILE_SIZE);
        
        // Map the normalized size to the configured size range
        return scaleOps.mapRange(normalizedSize, 0, 1, minSize, maxSize);
    }

    private updateAllNodeScales(): void {
        // Update all existing nodes with new scale based on current settings
        for (let i = 0; i < this.nodeInstances.count; i++) {
            this.nodeInstances.getMatrixAt(i, matrix);
            matrix.decompose(position, quaternion, scale);
            
            // Get the node ID for this instance
            const nodeId = this.getNodeId(i);
            if (!nodeId) continue;
            
            // Find the node data
            const node = Array.from(this.nodeIndices.entries())
                .find(([_, idx]) => idx === i)?.[0];
            if (!node) continue;

            // Calculate new scale
            const newScale = this.getNodeScale({ 
                id: nodeId, 
                data: { 
                    position: { x: position.x, y: position.y, z: position.z },
                    velocity: { x: 0, y: 0, z: 0 }
                }
            });
            scale.set(newScale, newScale, newScale);
            
            matrix.compose(position, quaternion, scale);
            this.nodeInstances.setMatrixAt(i, matrix);
            this.pendingUpdates.add(i);
        }
        
        if (this.pendingUpdates.size > 0) {
            this.nodeInstances.instanceMatrix.needsUpdate = true;
            this.pendingUpdates.clear();
        }
    }

    public updateNodePositions(updates: NodeUpdate[]): void {
        updates.forEach(update => {
            const index = this.nodeIndices.get(update.id);
            
            // Validate and clamp position
            position.set(update.position[0], update.position[1], update.position[2]);
            if (!this.validateVector3(position, this.MAX_POSITION)) {
                logger.warn(`Invalid position for node ${update.id}:`, position);
                this.clampVector3(position, this.MAX_POSITION);
            }

            // Validate and clamp velocity if present
            if (update.velocity) {
                velocity.set(update.velocity[0], update.velocity[1], update.velocity[2]);
                if (!this.validateVector3(velocity, this.MAX_VELOCITY)) {
                    logger.warn(`Invalid velocity for node ${update.id}:`, velocity);
                    this.clampVector3(velocity, this.MAX_VELOCITY);
                }
            }

            if (index === undefined) {
                // New node
                const newIndex = this.nodeInstances.count;
                if (newIndex < MAX_INSTANCES) {
                    this.nodeIndices.set(update.id, newIndex);
                    this.nodeInstances.count++;

                    // Calculate scale based on node properties
                    const scaleValue = this.getNodeScale({
                        id: update.id,
                        data: {
                            position: { x: position.x, y: position.y, z: position.z },
                            velocity: { x: 0, y: 0, z: 0 },
                            metadata: update.metadata
                        }
                    });
                    scale.set(scaleValue, scaleValue, scaleValue);
                    
                    if (update.velocity) {
                        const vel = new Vector3(update.velocity[0], update.velocity[1], update.velocity[2]);
                        this.velocities.set(newIndex, vel);
                    }
                    
                    matrix.compose(position, quaternion, scale);
                    this.nodeInstances.setMatrixAt(newIndex, matrix);
                    this.nodeInstances.setColorAt(newIndex, VISIBLE);
                    
                    this.pendingUpdates.add(newIndex);
                    logger.debug(`Added new node at index ${newIndex}`);
                } else {
                    logger.warn('Maximum instance count reached, cannot add more nodes');
                }
                return;
            }

            // Update existing node
            if (update.velocity) {
                this.velocities.set(index, velocity.clone());
            }
            
            // Calculate scale based on node properties
            const scaleValue = this.getNodeScale({
                id: update.id,
                data: {
                    position: { x: position.x, y: position.y, z: position.z },
                    velocity: { x: 0, y: 0, z: 0 },
                    metadata: update.metadata
                }
            });
            scale.set(scaleValue, scaleValue, scaleValue);
            
            matrix.compose(position, quaternion, scale);
            this.nodeInstances.setMatrixAt(index, matrix);
            this.pendingUpdates.add(index);
        });

        if (this.pendingUpdates.size > 0) {
            this.nodeInstances.instanceMatrix.needsUpdate = true;
            this.pendingUpdates.clear();
        }
    }

    public update(camera: Camera, passedDeltaTime?: number): void {
        this.frameCount++;
        
        // Update positions based on velocity
        const currentTime = performance.now();
        const deltaTime = passedDeltaTime !== undefined ? 
            passedDeltaTime : 
            (currentTime - this.lastUpdateTime) / 1000; // Convert to seconds
        this.lastUpdateTime = currentTime;

        // Update positions based on velocities
        this.velocities.forEach((nodeVelocity, index) => {
            if (nodeVelocity.lengthSq() > 0) {
                this.nodeInstances.getMatrixAt(index, matrix);
                matrix.decompose(position, quaternion, scale);
                
                // Apply velocity
                velocity.copy(nodeVelocity).multiplyScalar(deltaTime);
                position.add(velocity);
                
                // Update matrix
                matrix.compose(position, quaternion, scale);
                this.nodeInstances.setMatrixAt(index, matrix);
                this.pendingUpdates.add(index);
            }
        });

        // Update visibility and LOD every N frames
        if (this.frameCount % VISIBILITY_UPDATE_INTERVAL === 0) {
            this.updateVisibilityAndLOD(camera);
        }

        if (this.pendingUpdates.size > 0) {
            this.nodeInstances.instanceMatrix.needsUpdate = true;
            this.pendingUpdates.clear();
        }
    }

    private updateVisibilityAndLOD(camera: Camera): void {
        const cameraPosition = camera.position;
        
        // Check each instance
        for (let i = 0; i < this.nodeInstances.count; i++) {
            this.nodeInstances.getMatrixAt(i, matrix);
            position.setFromMatrixPosition(matrix);
            
            const distance = position.distanceTo(cameraPosition);
            
            // Update geometry based on distance
            void this.geometryManager.getGeometryForDistance(distance);

            // Update visibility
            const visible = distance < this.geometryManager.getThresholdForLOD(LODLevel.LOW);
            this.nodeInstances.setColorAt(i, visible ? VISIBLE : INVISIBLE);
        }

        // Ensure updates are applied
        if (this.nodeInstances.instanceColor) {
            this.nodeInstances.instanceColor.needsUpdate = true;
        }
    }

    public dispose(): void {
        if (this.nodeInstances) {
            this.nodeInstances.geometry.dispose();
            this.scene.remove(this.nodeInstances);
        }
        this.nodeIndices.clear();
        this.pendingUpdates.clear();
        this.velocities.clear();
        NodeInstanceManager.instance = null!;
        logger.info('Disposed NodeInstanceManager');
    }

    public getInstanceMesh(): InstancedMesh {
        return this.nodeInstances;
    }

    public getNodeId(index: number): string | undefined {
        return Array.from(this.nodeIndices.entries())
            .find(([_, idx]) => idx === index)?.[0];
    }

    public getNodePosition(nodeId: string): Vector3 | undefined {
        const index = this.nodeIndices.get(nodeId);
        if (index !== undefined) {
            this.nodeInstances.getMatrixAt(index, matrix);
            const position = new Vector3();
            position.setFromMatrixPosition(matrix);
            return position;
        }
        return undefined;
    }
}