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
import { debugState } from '../../core/debugState';
import { createLogger, createErrorMetadata, createDataMetadata } from '../../core/logger';
import { UpdateThrottler } from '../../core/utils';

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
    private nodeIdToMetadataId: Map<string, string> = new Map();
    private tempVector = new Vector3();
    private metadataUpdateThrottler = new UpdateThrottler(100); // Update at most every 100ms
    private readonly MAX_POSITION = 1000.0; // Reasonable limit for safe positions

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
            logger.debug('XR mode status changed', createDataMetadata({ enabled }));
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
            logger.error('Failed to update settings:', createErrorMetadata(error));
        }
    }

    /**
     * Update node positions and states
     * @param nodes Array of node updates
     */
    public updateNodes(nodes: { id: string, data: NodeData }[]): void {
        if (!this.isInitialized) return;

        const shouldDebugLog = debugState.isEnabled() && debugState.isNodeDebugEnabled();
        
        // Create dedicated ID set to ensure unique handling
        const processedIds = new Set<string>();
        
        // Track node IDs and handle metadata mapping
        nodes.forEach((node, index) => {
            if (shouldDebugLog && index < 3) {
                // Log the first few nodes to help debug
                logger.debug(`Processing node ${index}: id=${node.id}, ` +
                             `metadataId=${(node as any).metadataId || 'undefined'}, ` +
                             `label=${(node as any).label || 'undefined'}`, 
                              createDataMetadata({
                                 hasMetadata: !!node.data.metadata,
                                 metadata: node.data.metadata,
                                 fileSize: node.data.metadata?.fileSize
                              }));
            }
            
            // Skip if this node ID has already been processed in this batch
            // This prevents duplicate processing which could lead to overwriting
            if (processedIds.has(node.id)) {
                logger.warn(`Skipping duplicate node ID: ${node.id}`);
                return;
            }
            processedIds.add(node.id);

            // Store the node ID in our index map
            this.nodeIndices.set(node.id, node.id);
            
            if (node.data.metadata) {
                // Extract the proper metadata name/ID from the node
                // This could be the file name without the .md extension
                let metadataId: string = node.id; // Default to node ID
                
                // First, try using any server-provided label
                if ('metadataId' in node && typeof node['metadataId'] === 'string') {
                    // Prefer explicit metadataId if available (this is the filename)
                    metadataId = node['metadataId'] as string;
                } else if ('label' in node && typeof node['label'] === 'string') {
                    metadataId = node['label'] as string;
                } else if ('metadata_id' in node && typeof node['metadata_id'] === 'string') {
                    // Next, check for a specific metadata_id property if it exists
                    metadataId = node['metadata_id'] as string;
                } else if (node.data.metadata.name) {
                    // Finally, fallback to metadata name
                    metadataId = node.data.metadata.name;
                }
                
                if (metadataId && metadataId !== node.id) {
                    this.nodeIdToMetadataId.set(node.id, metadataId);
                    this.metadataManager.mapNodeIdToMetadataId(node.id, metadataId);
                    if (shouldDebugLog) {
                        logger.debug(`Mapped node ID ${node.id} to metadata ID ${metadataId}`);
                    }
                } 
            }
            if (shouldDebugLog) {
                logger.debug('Tracking node', createDataMetadata({ nodeId: node.id }));
            }
        });

        // Update instance positions
        // Important: Use fresh map to avoid modifying the original nodes
        const nodePositionUpdates = nodes.map(node => ({
            // Extract just what's needed for position update
            id: node.id,
            metadata: node.data.metadata || {},
            position: node.data.position,
            velocity: node.data.velocity
        }));
        this.instanceManager.updateNodePositions(nodePositionUpdates);
       
        // Only update metadata if the throttler allows it
        if (this.metadataUpdateThrottler.shouldUpdate()) {
            // Update metadata for each node
            nodes.forEach(node => {
                if (node.data.metadata) {
                    // Ensure we have valid file size
                    const fileSize = node.data.metadata.fileSize && node.data.metadata.fileSize > 0 
                        ? node.data.metadata.fileSize 
                        : DEFAULT_FILE_SIZE;
                    
                    // Use the metadata ID from our mapping, or fall back to node.id
                    const metadataId = this.nodeIdToMetadataId.get(node.id) || node.id;
                    
                    if (shouldDebugLog) {
                        logger.debug('Updating node metadata', createDataMetadata({ 
                            nodeId: node.id, 
                            metadataId,
                            fileSize: node.data.metadata.fileSize,
                            hyperlinkCount: node.data.metadata.hyperlinkCount || 0
                        }));
                    }
                    
                    // Check if node has a label (from server)
                    let displayName: string = metadataId;
                    if ('metadataId' in node && typeof node['metadataId'] === 'string') {
                        displayName = node['metadataId'] as string;
                    } else if ('label' in node && typeof node['label'] === 'string') {
                        displayName = node['label'] as string;
                    } else if ('metadata_id' in node && typeof node['metadata_id'] === 'string') {
                        displayName = node['metadata_id'] as string;
                    } 
                    
                    // Make sure to map the node ID to the proper metadata ID for labels
                    if (displayName && displayName !== node.id) {
                        this.metadataManager.mapNodeIdToMetadataId(node.id, displayName);
                    }
                    
                    this.metadataManager.updateMetadata(node.id, {
                        id: node.id,
                        name: displayName || metadataId, // Use the best name available
                        position: node.data.position,
                        // Ensure proper metadata is set with appropriate defaults
                        commitAge: node.data.metadata.lastModified !== undefined 
                            ? node.data.metadata.lastModified 
                            : 0,
                        hyperlinkCount: (node.data.metadata.hyperlinkCount !== undefined && node.data.metadata.hyperlinkCount > 0)
                            ? node.data.metadata.hyperlinkCount
                            : node.data.metadata.links?.length || 0,
                        importance: node.data.metadata.hyperlinkCount || 0,
                        fileSize: fileSize, // Use the provided fileSize
                        nodeSize: this.calculateNodeSize(fileSize),
                    });
                }
            });
        }
    }
 
    /**
     * Validates and fixes a Vector3 if it contains NaN or infinite values
     * Returns true if the vector was valid, false if it needed correction
     */
    private validateAndFixVector3(vec: Vector3, label: string, nodeId: string): boolean {
        const isValid = !isNaN(vec.x) && !isNaN(vec.y) && !isNaN(vec.z) &&
                       isFinite(vec.x) && isFinite(vec.y) && isFinite(vec.z);
        
        if (!isValid) {
            // Log warning with details of the invalid values
            logger.warn(`Invalid ${label} values for node ${nodeId}`, createDataMetadata({
                x: vec.x,
                y: vec.y,
                z: vec.z,
                isNaNX: isNaN(vec.x),
                isNaNY: isNaN(vec.y),
                isNaNZ: isNaN(vec.z),
                isFiniteX: isFinite(vec.x),
                isFiniteY: isFinite(vec.y),
                isFiniteZ: isFinite(vec.z)
            }));
            
            // Fix the vector - replace NaN or infinite values with 0
            vec.x = isNaN(vec.x) || !isFinite(vec.x) ? 0 : vec.x;
            vec.y = isNaN(vec.y) || !isFinite(vec.y) ? 0 : vec.y;
            vec.z = isNaN(vec.z) || !isFinite(vec.z) ? 0 : vec.z;
            
            // Also clamp to reasonable bounds
            vec.x = Math.max(-this.MAX_POSITION, Math.min(this.MAX_POSITION, vec.x));
            vec.y = Math.max(-this.MAX_POSITION, Math.min(this.MAX_POSITION, vec.y));
            vec.z = Math.max(-this.MAX_POSITION, Math.min(this.MAX_POSITION, vec.z));
        }
        return isValid;
    }

    public updateNodePositions(nodes: { 
        id: string, 
        data: { 
            position: Vector3,
            velocity: Vector3
        } 
    }[]): void {
        if (!this.isInitialized) return;

        try {
            // Handle NaN values and validate positions before passing to instance manager
            const validatedNodes = nodes.map(node => {
                this.validateAndFixVector3(node.data.position, 'position', node.id);
                if (node.data.velocity) this.validateAndFixVector3(node.data.velocity, 'velocity', node.id);
                return { id: node.id, position: node.data.position, velocity: node.data.velocity };
            });
            
            this.instanceManager.updateNodePositions(validatedNodes);
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
            this.nodeIndices.forEach((id) => {
                    const position = this.instanceManager.getNodePosition(id);
                    if (position) {
                        this.tempVector.copy(position);
                        
                        // Calculate dynamic offset based on node size
                        // Use the node's calculated size for offset
                        const nodeSize = this.calculateNodeSize();
                        
                        this.tempVector.y += nodeSize * 0.03; // Drastically reduced offset for much closer label positioning
                        // Update individual label position
                        this.metadataManager.updatePosition(id, this.tempVector.clone());
                    }
            });
            this.frameCount++;
        } catch (error) {
            logger.error('Error updating metadata positions:', createErrorMetadata(error));
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
            this.nodeIdToMetadataId.clear();
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

    /**
     * Get the metadata ID for a given node ID
     * This is useful for retrieving the human-readable name (file name)
     * @param nodeId Node ID to look up
     * @returns Metadata ID (filename) or undefined if not found
     */
    public getMetadataId(nodeId: string): string | undefined {
        return this.nodeIdToMetadataId.get(nodeId);
    }
}