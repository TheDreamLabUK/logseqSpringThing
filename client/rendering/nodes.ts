import * as THREE from 'three';
import { Node } from '../core/types';
import { createLogger } from '../core/logger';
import { settingsManager } from '../state/settings';
import type { Settings } from '../types/settings';
import { GeometryFactory } from './factories/GeometryFactory';
import { MaterialFactory } from './factories/MaterialFactory';
import { SettingsObserver } from '../state/SettingsObserver';

const logger = createLogger('NodeManager');

const FLOATS_PER_NODE = 6;  // x, y, z, vx, vy, vz

// Reusable objects for matrix calculations
const matrix = new THREE.Matrix4();
const quaternion = new THREE.Quaternion();
const position = new THREE.Vector3();
const scale = new THREE.Vector3(1, 1, 1);

// Batch size for matrix updates
const MATRIX_UPDATE_BATCH_SIZE = 1000;

export class NodeManager {
    private static instance: NodeManager;
    private currentSettings: Settings;
    private nodeInstances: THREE.InstancedMesh;
    private currentNodes: Node[] = [];
    private nodeIndices: Map<string, number> = new Map();
    private readonly materialFactory: MaterialFactory;
    private readonly geometryFactory: GeometryFactory;
    private readonly settingsObserver: SettingsObserver;
    
    // Matrix update queue for batching
    private pendingMatrixUpdates: Set<number> = new Set();
    private matrixUpdateScheduled: boolean = false;

    private getRenderContext(): 'ar' | 'desktop' {
        return this.currentSettings.xr.mode === 'immersive-ar' ? 'ar' : 'desktop';
    }

    private constructor() {
        this.currentSettings = settingsManager.getCurrentSettings();
        this.materialFactory = MaterialFactory.getInstance();
        this.geometryFactory = GeometryFactory.getInstance();
        this.settingsObserver = SettingsObserver.getInstance();

        const context = this.getRenderContext();

        // Create node instances with context-aware geometry and material
        this.nodeInstances = new THREE.InstancedMesh(
            this.geometryFactory.getNodeGeometry(this.currentSettings.xr.quality, context),
            this.materialFactory.getNodeMaterial(this.currentSettings, context),
            10000
        );

        // Set AR layer for node instances
        this.nodeInstances.layers.enable(1);
        this.nodeInstances.frustumCulled = true; // Enable frustum culling

        // Edge handling has been moved to EdgeManager

        this.setupSettingsSubscriptions();
    }

    private setupSettingsSubscriptions(): void {
        this.settingsObserver.subscribe('visualization', (_path: string, _value: any) => {
            const prevContext = this.getRenderContext();
            // Get fresh settings to ensure we have the complete state
            this.currentSettings = settingsManager.getCurrentSettings();
            const newContext = this.getRenderContext();

            // Update materials and geometry if context changed
            if (prevContext !== newContext) {
                if (this.nodeInstances) {
                    // Update node geometry and material
                    const nodeGeometry = this.geometryFactory.getNodeGeometry(
                        this.currentSettings.xr.quality,
                        newContext
                    );
                    const nodeMaterial = this.materialFactory.getNodeMaterial(
                        this.currentSettings,
                        newContext
                    );
                    this.nodeInstances.geometry.dispose();
                    this.nodeInstances.material.dispose();
                    this.nodeInstances.geometry = nodeGeometry;
                    this.nodeInstances.material = nodeMaterial;
                }

                // Edge handling has been moved to EdgeManager
            } else {
                // Just update material properties if context hasn't changed
                this.materialFactory.updateMaterial(`node-${newContext}`, this.currentSettings);
                this.materialFactory.updateMaterial(`edge-${newContext}`, this.currentSettings);
            }
        });

        // Subscribe to XR settings changes
        this.settingsObserver.subscribe('xr', (_path: string, _value: any) => {
            this.currentSettings = settingsManager.getCurrentSettings();
            const context = this.getRenderContext();
            this.materialFactory.updateMaterial(`node-${context}`, this.currentSettings);
            this.materialFactory.updateMaterial(`edge-${context}`, this.currentSettings);
        });
    }

    public static getInstance(): NodeManager {
        if (!NodeManager.instance) {
            NodeManager.instance = new NodeManager();
        }
        return NodeManager.instance;
    }

    private scheduleBatchUpdate(): void {
        if (this.matrixUpdateScheduled) return;
        this.matrixUpdateScheduled = true;

        requestAnimationFrame(() => {
            this.processBatchUpdate();
            this.matrixUpdateScheduled = false;
        });
    }

    private processBatchUpdate(): void {
        if (!this.nodeInstances || this.pendingMatrixUpdates.size === 0) return;

        let processed = 0;
        this.pendingMatrixUpdates.forEach(index => {
            if (processed >= MATRIX_UPDATE_BATCH_SIZE) {
                return; // Process remaining updates in next batch
            }

            const node = this.currentNodes[index];
            if (!node) return;

            position.set(
                node.data.position.x,
                node.data.position.y + 1.5,
                node.data.position.z
            );

            const baseSize = this.currentSettings.visualization.nodes.baseSize || 1;
            scale.set(baseSize, baseSize, baseSize);

            matrix.compose(position, quaternion, scale);
            this.nodeInstances.setMatrixAt(index, matrix);

            processed++;
            this.pendingMatrixUpdates.delete(index);
        });

        if (processed > 0) {
            this.nodeInstances.instanceMatrix.needsUpdate = true;
        }

        if (this.pendingMatrixUpdates.size > 0) {
            this.scheduleBatchUpdate(); // Schedule next batch if needed
        }
    }

    public updatePositions(positions: Float32Array): void {
        if (!this.nodeInstances) return;

        const count = Math.min(positions.length / FLOATS_PER_NODE, this.nodeInstances.count);
        
        for (let i = 0; i < count; i++) {
            const baseIndex = i * FLOATS_PER_NODE;
            
            const x = positions[baseIndex];
            const y = positions[baseIndex + 1];
            const z = positions[baseIndex + 2];
            
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) ||
                Math.abs(x) > 1000 || Math.abs(y) > 1000 || Math.abs(z) > 1000) {
                logger.warn(`Skipping invalid position for node ${i}: (${x}, ${y}, ${z})`);
                continue;
            }

            this.pendingMatrixUpdates.add(i);
        }

        this.scheduleBatchUpdate();
    }

    public getAllNodeMeshes(): THREE.InstancedMesh[] {
        return [this.nodeInstances];
    }

    public getNodePosition(nodeId: string): THREE.Vector3 {
        const node = this.currentNodes.find(n => n.id === nodeId);
        if (!node) {
            throw new Error(`Node ${nodeId} not found`);
        }
        return new THREE.Vector3(
            node.data.position.x,
            node.data.position.y,
            node.data.position.z
        );
    }

    public updateNodePosition(nodeId: string, newPosition: THREE.Vector3): void {
        const index = this.nodeIndices.get(nodeId);
        if (index === undefined) {
            throw new Error(`Node ${nodeId} not found`);
        }

        const node = this.currentNodes[index];
        if (node) {
            node.data.position = {
                x: newPosition.x,
                y: newPosition.y,
                z: newPosition.z
            };
            this.pendingMatrixUpdates.add(index);
            this.scheduleBatchUpdate();
        }
    }

    public getCurrentNodes(): Node[] {
        return [...this.currentNodes];
    }

    public updateNodes(nodes: Node[]): void {
        this.currentNodes = nodes;
        const positions = new Float32Array(nodes.length * FLOATS_PER_NODE);
        
        nodes.forEach((node, index) => {
            const baseIndex = index * FLOATS_PER_NODE;
            positions[baseIndex] = node.data.position.x;
            positions[baseIndex + 1] = node.data.position.y + 1.5;
            positions[baseIndex + 2] = node.data.position.z;
            positions[baseIndex + 3] = 0;
            positions[baseIndex + 4] = 0;
            positions[baseIndex + 5] = 0;
            
            this.nodeIndices.set(node.id, index);
        });
        
        this.updatePositions(positions);
    }

    public dispose(): void {
        if (this.nodeInstances) {
            this.nodeInstances.geometry.dispose();
            this.nodeInstances.material.dispose();
        }
        // Edge cleanup has been moved to EdgeManager
        this.pendingMatrixUpdates.clear();
    }
}
