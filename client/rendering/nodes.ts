import * as THREE from 'three';
import { Node } from '../core/types';
import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';
import type { Settings } from '../types/settings';
import type { NodeSettings, PhysicsSettings } from '../core/types';

const logger = createLogger('NodeManager');

const NODE_SEGMENTS = 32;
const EDGE_SEGMENTS = 8;
const BINARY_VERSION = 1.0;
const FLOATS_PER_NODE = 6;  // x, y, z, vx, vy, vz
const VERSION_OFFSET = 1;    // Skip version float

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
            new THREE.SphereGeometry(1, NODE_SEGMENTS, NODE_SEGMENTS),
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

        const nodeGeometry = new THREE.SphereGeometry(1, NODE_SEGMENTS, NODE_SEGMENTS);
        const edgeGeometry = new THREE.CylinderGeometry(0.1, 0.1, 1, EDGE_SEGMENTS);
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

    public updatePositions(floatArray: Float32Array): void {
        try {
            const version = floatArray[0];
            if (version !== BINARY_VERSION) {
                logger.warn(`Received binary data version ${version}, expected ${BINARY_VERSION}`);
                return;
            }

            const nodeCount = Math.floor((floatArray.length - VERSION_OFFSET) / FLOATS_PER_NODE);
            if (nodeCount > this.currentNodes.length) {
                logger.warn(`Received more nodes than currently tracked: ${nodeCount} > ${this.currentNodes.length}`);
                return;
            }

            for (let i = 0; i < nodeCount; i++) {
                const baseIndex = VERSION_OFFSET + (i * FLOATS_PER_NODE);
                position.set(
                    floatArray[baseIndex],
                    floatArray[baseIndex + 1],
                    floatArray[baseIndex + 2]
                );

                matrix.compose(position, quaternion, scale);
                this.nodeInstances.setMatrixAt(i, matrix);

                const node = this.currentNodes[i];
                if (node) {
                    node.data.position.x = floatArray[baseIndex];
                    node.data.position.y = floatArray[baseIndex + 1];
                    node.data.position.z = floatArray[baseIndex + 2];
                }
            }

            this.nodeInstances.instanceMatrix.needsUpdate = true;
        } catch (error) {
            logger.error('Error updating positions:', error);
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
