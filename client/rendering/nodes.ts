import * as THREE from 'three';
import { Node } from '../core/types';
import { createLogger } from '../core/logger';
import { settingsManager } from '../state/settings';
import type { Settings } from '../types/settings';
import type { NodeSettings, PhysicsSettings } from '../core/types';
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
            this.geometryFactory.getNodeGeometry('high'),
            this.material
        );

        this.setupSettingsSubscriptions();
    }

    public handleSettingChange(setting: keyof NodeSettings, value: any): void {
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

    public handlePhysicsSettingChange(setting: keyof PhysicsSettings, value: any): void {
        // Dummy implementation for now
        logger.debug(`Physics setting change: ${String(setting)} = ${value}`);
    }

    private setupSettingsSubscriptions(): void {
        this.settingsObserver.subscribe('NodeRenderer', (settings) => {
            this.currentSettings = settings;
            Object.keys(settings.nodes).forEach(setting => {
                this.handleSettingChange(setting as keyof NodeSettings, settings.nodes[setting as keyof NodeSettings]);
            });
            Object.keys(settings.physics).forEach(setting => {
                this.handlePhysicsSettingChange(setting as keyof PhysicsSettings, settings.physics[setting as keyof PhysicsSettings]);
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
            this.geometryFactory.getNodeGeometry('high'),
            this.nodeRenderer.material,
            10000
        );

        this.edgeInstances = new THREE.InstancedMesh(
            this.geometryFactory.getHologramGeometry('ring', 'medium'),
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
            
            // Update position
            position.set(
                positions[baseIndex],
                positions[baseIndex + 1],
                positions[baseIndex + 2]
            );
            
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
