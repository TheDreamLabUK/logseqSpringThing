import {
    Scene,
    Camera,
    Material,
    InstancedMesh,
    Vector3
} from 'three';
import { NodeGeometryManager } from './geometry/NodeGeometryManager';
import { NodeInstanceManager } from './instance/NodeInstanceManager';
import { NodeMetadataManager } from './metadata/NodeMetadataManager';
import { NodeInteractionManager } from './interaction/NodeInteractionManager';
import { NodeManagerInterface, NodeManagerError, NodeManagerErrorType } from './NodeManagerInterface';
import { NodeData } from '../../core/types';
import { XRHandWithHaptics } from '../../types/xr';
import { createLogger } from '../../core/logger';

const logger = createLogger('NodeManagerFacade');

// Constants for size calculation
const DEFAULT_FILE_SIZE = 1000; // 1KB default
const MAX_FILE_SIZE = 10485760; // 10MB max for scaling
const MIN_NODE_SIZE = 0;
const MAX_NODE_SIZE = 50;

/**
 * NodeManagerFacade provides a unified interface to the node management system.
 * It coordinates between the geometry, instance, metadata, and interaction managers.
 */
export class NodeManagerFacade implements NodeManagerInterface {
    private static instance: NodeManagerFacade;
    private camera: Camera;
    private geometryManager: NodeGeometryManager;
    private instanceManager: NodeInstanceManager;
    private metadataManager: NodeMetadataManager;
    private interactionManager: NodeInteractionManager;
    private isInitialized: boolean = false;
    private frameCount: number = 0;
    private nodeIndices: Map<string, string> = new Map();
    private tempVector = new Vector3();

    private constructor(scene: Scene, camera: Camera, material: Material) {
        this.camera = camera;

        try {
            // Initialize managers in the correct order
            this.geometryManager = NodeGeometryManager.getInstance();
            this.instanceManager = NodeInstanceManager.getInstance(scene, material);
            this.metadataManager = NodeMetadataManager.getInstance(scene);
            
            // Initialize interaction manager with instance mesh
            const instanceMesh = this.instanceManager.getInstanceMesh();
            this.interactionManager = NodeInteractionManager.getInstance(instanceMesh);

            this.isInitialized = true;
            logger.info('NodeManagerFacade initialized');
        } catch (error) {
            throw new NodeManagerError(
                NodeManagerErrorType.INITIALIZATION_FAILED,
                'Failed to initialize NodeManagerFacade',
                error
            );
        }
    }
    
    public static getInstance(scene: Scene, camera: Camera, material: Material): NodeManagerFacade {
        if (!NodeManagerFacade.instance) {
            NodeManagerFacade.instance = new NodeManagerFacade(scene, camera, material);
        }
        return NodeManagerFacade.instance;
    }

    private calculateNodeSize(fileSize: number = DEFAULT_FILE_SIZE): number {
        // Map file size logarithmically to 0-1 range
        const normalizedSize = Math.log(Math.min(fileSize, MAX_FILE_SIZE)) / Math.log(MAX_FILE_SIZE);
        // Map to metadata node size range (0-50)
        return MIN_NODE_SIZE + normalizedSize * (MAX_NODE_SIZE - MIN_NODE_SIZE);
    }

    public setXRMode(enabled: boolean): void {
        if (!this.isInitialized) return;

        try {
            const instanceMesh = this.instanceManager.getInstanceMesh();
            instanceMesh.layers.set(enabled ? 1 : 0);
            this.metadataManager.setXRMode(enabled);
            logger.debug(`XR mode ${enabled ? 'enabled' : 'disabled'}`);
        } catch (error) {
            throw new NodeManagerError(
                NodeManagerErrorType.XR_MODE_SWITCH_FAILED,
                'Failed to switch XR mode',
                error
            );
        }
    }

    public handleSettingsUpdate(settings: any): void {
        if (!this.isInitialized) return;

        try {
            // Update metadata visibility threshold if needed
            if (settings.visualization?.labels?.visibilityThreshold) {
                this.metadataManager.updateVisibilityThreshold(
                    settings.visualization.labels.visibilityThreshold
                );
            }
        } catch (error) {
            logger.error('Failed to update settings:', error);
        }
    }

    /**
     * Update node positions and states
     * @param nodes Array of node updates
     */
    public updateNodes(nodes: { id: string, data: NodeData }[]): void {
        if (!this.isInitialized) return;

        // Track node IDs
        nodes.forEach(node => {
            this.nodeIndices.set(node.id, node.id);
            logger.debug(`Tracking node ${node.id}`);
        });

        // Update instance positions
        this.instanceManager.updateNodePositions(nodes.map(node => ({
            id: node.id,
            metadata: node.data.metadata,
            position: [
                node.data.position.x,
                node.data.position.y,
                node.data.position.z
            ],
            velocity: node.data.velocity ? [
                node.data.velocity.x,
                node.data.velocity.y,
                node.data.velocity.z
            ] : undefined
        })));

        // Update metadata for each node
        nodes.forEach(node => {
            if (node.data.metadata) {
                const fileSize = node.data.metadata.fileSize || DEFAULT_FILE_SIZE;
                logger.debug(`Updating metadata for node ${node.id}`);
                this.metadataManager.updateMetadata(node.id, {
                    id: node.id,
                    name: node.data.metadata.name || '',
                    position: node.data.position,
                    commitAge: 0,
                    hyperlinkCount: node.data.metadata.links?.length || 0,
                    importance: 0,
                    fileSize: fileSize,
                    nodeSize: this.calculateNodeSize(fileSize)
                });
            }
        });
    }

    public updateNodePositions(nodes: { 
        id: string, 
        data: { 
            position: [number, number, number],
            velocity?: [number, number, number]
        } 
    }[]): void {
        if (!this.isInitialized) return;

        try {
            // Update instance positions
            this.instanceManager.updateNodePositions(nodes.map(node => ({
                id: node.id,
                position: node.data.position,
                velocity: node.data.velocity
            })));
        } catch (error) {
            throw new NodeManagerError(
                NodeManagerErrorType.UPDATE_FAILED,
                'Failed to update node positions',
                error
            );
        }
    }

    /**
     * Handle XR hand interactions
     * @param hand XR hand data with haptic feedback
     */
    public handleHandInteraction(hand: XRHandWithHaptics): void {
        if (!this.isInitialized) return;
        this.interactionManager.handleHandInteraction(hand);
    }

    /**
     * Update the visualization state
     * @param deltaTime Time since last update
     */
    public update(deltaTime: number): void {
        if (!this.isInitialized) return;

        // Update instance visibility and LOD
        this.instanceManager.update(this.camera, deltaTime);

        // Update metadata positions to match instances
        try {
            // Only update positions every few frames for performance
            if (this.frameCount % 5 === 0) {
                this.nodeIndices.forEach((id) => {
                    const position = this.instanceManager.getNodePosition(id);
                    if (position) {
                        this.tempVector.copy(position);
                        this.tempVector.y += 1.5; // Offset above node
                        // Update individual label position
                        this.metadataManager.updatePosition(id, this.tempVector);
                    }
                });
                logger.debug('Updated metadata positions');
            }
            this.frameCount++;
        } catch (error) {
            logger.error('Error updating metadata positions:', error);
        }

        // Update metadata labels
        this.metadataManager.update(this.camera);
    }

    /**
     * Clean up resources
     */
    public dispose(): void {
        if (!this.isInitialized) return;

        try {
            this.geometryManager.dispose();
            this.instanceManager.dispose();
            this.metadataManager.dispose();
            this.interactionManager.dispose();
            this.nodeIndices.clear();

            NodeManagerFacade.instance = null!;
            this.isInitialized = false;
            logger.info('NodeManagerFacade disposed');
        } catch (error) {
            throw new NodeManagerError(
                NodeManagerErrorType.RESOURCE_CLEANUP_FAILED,
                'Failed to dispose NodeManagerFacade',
                error
            );
        }
    }

    /**
     * Get the underlying InstancedMesh
     * Useful for adding to scenes or handling special cases
     */
    public getInstancedMesh(): InstancedMesh {
        return this.instanceManager.getInstanceMesh();
    }

    /**
     * Get node ID from instance index
     * @param index Instance index in the InstancedMesh
     * @returns Node ID or undefined if not found
     */
    public getNodeId(index: number): string | undefined {
        return this.instanceManager.getNodeId(index);
    }

    /**
     * Get the underlying NodeInstanceManager
     * @returns The NodeInstanceManager instance
     */
    public getNodeInstanceManager(): NodeInstanceManager {
        return this.instanceManager;
    }
}