import * as THREE from 'three';
import { Node } from '../core/types';
import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';
import type { Settings } from '../types/settings';
import type { NodeSettings, PhysicsSettings } from '../core/types';

const logger = createLogger('NodeManager');

const FLOATS_PER_NODE = 6;  // x, y, z, vx, vy, vz

// Reusable vectors and matrices
const matrix = new THREE.Matrix4();
const quaternion = new THREE.Quaternion();
const position = new THREE.Vector3();
const scale = new THREE.Vector3(1, 1, 1);

export class NodeRenderer {
    public readonly material: THREE.Material;
    protected currentSettings: Settings;
    public mesh: THREE.Mesh;

    constructor() {
        this.currentSettings = settingsManager.getCurrentSettings();
        this.material = new THREE.MeshPhongMaterial({
            color: 0x4fc3f7,
            shininess: 30,
            specular: 0x004ba0,
            transparent: true,
            opacity: 0.9,
        });

        this.mesh = new THREE.Mesh(
            new THREE.SphereGeometry(1, 32, 32),
            this.material
        );

        this.setupSettingsSubscriptions();
    }

    public handleSettingChange(setting: keyof NodeSettings, value: any): void {
        try {
            switch (setting) {
                case 'baseColor':
                    this.material.color.set(value as string);
                    break;
                case 'opacity':
                    this.material.opacity = value as number;
                    break;
                case 'baseSize':
                    this.mesh.scale.set(value, value, value);
                    break;
                default:
                    // Other settings handled elsewhere
                    break;
            }
            (this.material as any).needsUpdate = true;
        } catch (error) {
            logger.error(`Error applying node setting change for ${String(setting)}:`, error);
        }
    }

    public handlePhysicsSettingChange(setting: keyof PhysicsSettings, value: any): void {
        // Dummy implementation for now
        logger.debug(`Physics setting change: ${String(setting)} = ${value}`);
    }

    private setupSettingsSubscriptions(): void {
        Object.keys(this.currentSettings.nodes).forEach(setting => {
            settingsManager.subscribe('nodes', setting as keyof NodeSettings, (value) => {
                this.handleSettingChange(setting as keyof NodeSettings, value);
            });
        });

        Object.keys(this.currentSettings.physics).forEach(setting => {
            settingsManager.subscribe('physics', setting as keyof PhysicsSettings, (value) => {
                this.handlePhysicsSettingChange(setting as keyof PhysicsSettings, value);
            });
        });
    }
}

export class NodeManager {
    private static instance: NodeManager;
    private currentSettings: Settings;
    private nodeInstances: THREE.InstancedMesh;
    private edgeInstances: THREE.InstancedMesh;
    private unsubscribers: Array<() => void> = [];
    private nodeRenderer: NodeRenderer;
    private currentNodes: Node[] = [];
    private nodeIndices: Map<string, number> = new Map();

    private constructor() {
        this.currentSettings = settingsManager.getCurrentSettings();
        this.nodeRenderer = new NodeRenderer();

        const nodeGeometry = new THREE.SphereGeometry(1, 32, 32);
        const edgeGeometry = new THREE.CylinderGeometry(0.1, 0.1, 1, 8);
        edgeGeometry.rotateX(Math.PI / 2);

        this.nodeInstances = new THREE.InstancedMesh(
            nodeGeometry,
            this.nodeRenderer.material,
            10000
        );

        this.edgeInstances = new THREE.InstancedMesh(
            edgeGeometry,
            this.createEdgeMaterial(),
            30000
        );

        this.setupSettingsSubscriptions();
    }

    private createEdgeMaterial(): THREE.Material {
        return new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: this.currentSettings.edges.opacity,
            color: this.currentSettings.nodes.baseColor,
        });
    }

    public updatePositions(positions: Float32Array): void {
        if (!this.nodeInstances) return;

        const count = Math.min(positions.length / FLOATS_PER_NODE, this.nodeInstances.count);
        
        for (let i = 0; i < count; i++) {
            const baseIndex = i * FLOATS_PER_NODE;
            
            // Get position values
            const x = positions[baseIndex];
            const y = positions[baseIndex + 1];
            const z = positions[baseIndex + 2];
            
            // Skip invalid positions
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) ||
                Math.abs(x) > 1000 || Math.abs(y) > 1000 || Math.abs(z) > 1000) {
                logger.warn(`Skipping invalid position for node ${i}: (${x}, ${y}, ${z})`);
                continue;
            }
            
            // Update position
            position.set(x, y, z);
            
            // Set initial scale based on settings
            const baseSize = this.currentSettings.nodes.baseSize || 1;
            scale.set(baseSize, baseSize, baseSize);
            
            // Update instance matrix
            matrix.compose(position, quaternion, scale);
            this.nodeInstances.setMatrixAt(i, matrix);
        }
        
        this.nodeInstances.instanceMatrix.needsUpdate = true;
        
        // Force a render update
        if (this.currentSettings.animations.enableNodeAnimations) {
            requestAnimationFrame(() => {
                this.nodeInstances.instanceMatrix.needsUpdate = true;
            });
        }
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
            node.data.position = { x: newPosition.x, y: newPosition.y, z: newPosition.z };
        }

        matrix.compose(newPosition, quaternion, scale);
        this.nodeInstances.setMatrixAt(index, matrix);
        this.nodeInstances.instanceMatrix.needsUpdate = true;
    }

    private setupSettingsSubscriptions(): void {
        Object.keys(this.currentSettings.nodes).forEach(setting => {
            const unsubscribe = settingsManager.subscribe('nodes', setting as keyof NodeSettings, (value) => {
                this.nodeRenderer.handleSettingChange(setting as keyof NodeSettings, value);
            });
            this.unsubscribers.push(unsubscribe);
        });

        Object.keys(this.currentSettings.physics).forEach(setting => {
            const unsubscribe = settingsManager.subscribe('physics', setting as keyof PhysicsSettings, (value) => {
                this.nodeRenderer.handlePhysicsSettingChange(setting as keyof PhysicsSettings, value);
            });
            this.unsubscribers.push(unsubscribe);
        });
    }

    public static getInstance(): NodeManager {
        if (!NodeManager.instance) {
            NodeManager.instance = new NodeManager();
        }
        return NodeManager.instance;
    }

    public dispose(): void {
        if (this.nodeInstances) {
            if (this.nodeInstances.geometry) {
                this.nodeInstances.geometry.dispose();
            }
            if (this.nodeInstances.material instanceof THREE.Material) {
                this.nodeInstances.material.dispose();
            }
            this.nodeInstances.dispose();
        }
        if (this.edgeInstances) {
            if (this.edgeInstances.geometry) {
                this.edgeInstances.geometry.dispose();
            }
            if (this.edgeInstances.material instanceof THREE.Material) {
                this.edgeInstances.material.dispose();
            }
            this.edgeInstances.dispose();
        }
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
