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

// Reusable vectors and matrices
const matrix = new THREE.Matrix4();
const quaternion = new THREE.Quaternion();
const position = new THREE.Vector3();
const scale = new THREE.Vector3(1, 1, 1);

export class NodeRenderer {
    public readonly material: THREE.Material;
    protected currentSettings: Settings;
    public mesh: THREE.Mesh;
    private readonly materialFactory: MaterialFactory;
    private readonly geometryFactory: GeometryFactory;
    private readonly settingsObserver: SettingsObserver;

    constructor() {
        this.currentSettings = settingsManager.getCurrentSettings();
        this.materialFactory = MaterialFactory.getInstance();
        this.geometryFactory = GeometryFactory.getInstance();
        this.settingsObserver = SettingsObserver.getInstance();

        this.material = this.materialFactory.getPhongNodeMaterial();
        this.mesh = new THREE.Mesh(
            this.geometryFactory.getNodeGeometry(this.currentSettings.xr.quality),
            this.material
        );

        this.setupSettingsSubscriptions();
    }

    public handleSettingChange(setting: keyof Settings['visualization']['nodes'], value: any): void {
        try {
            switch (setting) {
                case 'baseColor':
                case 'opacity':
                    this.materialFactory.updateMaterial('node-phong', this.currentSettings);
                    break;
                case 'baseSize':
                    this.mesh.scale.set(value, value, value);
                    break;
                default:
                    // Other settings handled elsewhere
                    break;
            }
        } catch (error) {
            logger.error(`Error applying node setting change for ${String(setting)}:`, error);
        }
    }

    public handlePhysicsSettingChange(setting: keyof Settings['visualization']['physics'], value: any): void {
        // Dummy implementation for now
        logger.debug(`Physics setting change: ${String(setting)} = ${value}`);
    }

    private setupSettingsSubscriptions(): void {
        this.settingsObserver.subscribe('NodeRenderer', (settings) => {
            this.currentSettings = settings;
            Object.keys(settings.visualization.nodes).forEach(setting => {
                this.handleSettingChange(
                    setting as keyof Settings['visualization']['nodes'],
                    settings.visualization.nodes[setting as keyof Settings['visualization']['nodes']]
                );
            });
            Object.keys(settings.visualization.physics).forEach(setting => {
                this.handlePhysicsSettingChange(
                    setting as keyof Settings['visualization']['physics'],
                    settings.visualization.physics[setting as keyof Settings['visualization']['physics']]
                );
            });
        });
    }
}

export class NodeManager {
    private static instance: NodeManager;
    private currentSettings: Settings;
    private nodeInstances: THREE.InstancedMesh;
    private edgeInstances: THREE.InstancedMesh;
    private nodeRenderer: NodeRenderer;
    private currentNodes: Node[] = [];
    private nodeIndices: Map<string, number> = new Map();
    private readonly materialFactory: MaterialFactory;
    private readonly geometryFactory: GeometryFactory;
    private readonly settingsObserver: SettingsObserver;

    private constructor() {
        this.currentSettings = settingsManager.getCurrentSettings();
        this.materialFactory = MaterialFactory.getInstance();
        this.geometryFactory = GeometryFactory.getInstance();
        this.settingsObserver = SettingsObserver.getInstance();
        this.nodeRenderer = new NodeRenderer();

        this.nodeInstances = new THREE.InstancedMesh(
            this.geometryFactory.getNodeGeometry(this.currentSettings.xr.quality),
            this.nodeRenderer.material,
            10000
        );

        this.edgeInstances = new THREE.InstancedMesh(
            this.geometryFactory.getHologramGeometry('ring', this.currentSettings.xr.quality),
            this.materialFactory.getMetadataMaterial(),
            30000
        );

        this.setupSettingsSubscriptions();
    }

    private setupSettingsSubscriptions(): void {
        this.settingsObserver.subscribe('NodeManager', (settings) => {
            this.currentSettings = settings;
            this.materialFactory.updateMaterial('metadata', settings);
        });
    }

    public static getInstance(): NodeManager {
        if (!NodeManager.instance) {
            NodeManager.instance = new NodeManager();
        }
        return NodeManager.instance;
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
            const baseSize = this.currentSettings.visualization.nodes.baseSize || 1;
            scale.set(baseSize, baseSize, baseSize);
            
            // Update instance matrix
            matrix.compose(position, quaternion, scale);
            this.nodeInstances.setMatrixAt(i, matrix);
        }
        
        this.nodeInstances.instanceMatrix.needsUpdate = true;
        
        // Force a render update
        if (this.currentSettings.visualization.animations.enableNodeAnimations) {
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
            node.data.position = {
                x: newPosition.x,
                y: newPosition.y,
                z: newPosition.z
            };

            matrix.compose(newPosition, quaternion, scale);
            this.nodeInstances.setMatrixAt(index, matrix);
            this.nodeInstances.instanceMatrix.needsUpdate = true;
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
            positions[baseIndex + 1] = node.data.position.y;
            positions[baseIndex + 2] = node.data.position.z;
            // Velocity components (if needed)
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
        if (this.edgeInstances) {
            this.edgeInstances.geometry.dispose();
            this.edgeInstances.material.dispose();
        }
    }
}
