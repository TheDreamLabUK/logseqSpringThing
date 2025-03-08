import {
    Group,
    Scene,
    PerspectiveCamera,
    Vector3,
    Color,
    Object3D,
    SphereGeometry,
    MeshBasicMaterial,
    Mesh
} from 'three';
import { NodeMetadata } from '../types/metadata';
import { Settings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
import { createLogger, Logger } from '../core/logger';
import { debugState } from '../core/debugState';
import { UnifiedTextRenderer } from './UnifiedTextRenderer';

interface MetadataLabelGroup extends Group {
    name: string;
    userData: {
        isMetadata: boolean;
        nodeId?: string;
    };
}

export type MetadataLabelCallback = (group: MetadataLabelGroup) => void;

export class MetadataVisualizer {
    private scene: Scene;
    private labelGroup: Group;
    private settings: Settings;
    private debugEnabled: boolean = false;
    private textRenderer: UnifiedTextRenderer;
    private metadataGroups: Map<string, MetadataLabelGroup>;
    private logger: Logger;
    private debugHelpers: Map<string, Object3D>;
    private labelUpdateCount: number = 0;
    private lastClearTime: number = 0;
    private visibilityThreshold: number = 50; // Default visibility threshold

    constructor(camera: PerspectiveCamera, scene: Scene, settings: Settings) {
        this.scene = scene;
        this.settings = settings;
        this.metadataGroups = new Map();
        this.logger = createLogger('MetadataVisualizer');
        this.debugEnabled = debugState.isEnabled();
        
        this.debugHelpers = new Map();
        this.visibilityThreshold = settings.visualization.labels.visibilityThreshold || 50;
        
        // On initialization, log our settings
        if (this.debugEnabled) {
            console.log('[MetadataVisualizer] Initialized with settings:', {
                enableLabels: settings.visualization.labels.enableLabels,
                textColor: settings.visualization.labels.textColor,
                desktopFontSize: settings.visualization.labels.desktopFontSize,
                visibilityThreshold: this.visibilityThreshold
            });
        }
        
        this.logger.info('Initializing MetadataVisualizer with settings:', {
            enableLabels: settings.visualization.labels.enableLabels,
            textColor: settings.visualization.labels.textColor,
            desktopFontSize: settings.visualization.labels.desktopFontSize,
            visibilityThreshold: this.visibilityThreshold,
            textOutlineColor: settings.visualization.labels.textOutlineColor,
            billboardMode: settings.visualization.labels.billboardMode
        });

        this.labelGroup = new Group();
        this.scene.add(this.labelGroup);
        
        // Initialize text renderer
        this.textRenderer = new UnifiedTextRenderer(camera, scene, settings.visualization.labels);
        
        // Enable both layers by default for desktop mode
        this.labelGroup.layers.enable(0);
        this.labelGroup.layers.enable(1);
        
        // Set initial layer mode
        this.setXRMode(platformManager.isXRMode);
        
        // Listen for XR mode changes
        platformManager.on('xrmodechange', (enabled: boolean) => {
            this.setXRMode(enabled);
        });
    }

    public async createMetadataLabel(metadata: NodeMetadata, nodeId: string): Promise<MetadataLabelGroup> {
        // Track how many labels we've created
        this.labelUpdateCount++;

        // CRITICAL FIX: Ensure we're using the correct nodeId parameter passed in
        // The nodeId parameter from VisualizationController is now the correct numeric ID
        // that matches the binary WebSocket protocol IDs

        // Log detailed metadata info to help debug node label issues
        if (this.debugEnabled) {
            console.log(`[MetadataVisualizer] Creating label for node ${nodeId}:`, {
                name: metadata.name,
                fileSize: metadata.fileSize,
                hyperlinkCount: metadata.hyperlinkCount
            });
        }

        const group = new Group() as MetadataLabelGroup;
        group.name = 'metadata-label';
        group.userData = { 
            isMetadata: true,
            // Ensure we're storing the correct nodeId for position updates
            nodeId
        };


        // Format file size
        const fileSizeFormatted = !metadata.fileSize ? '0B' : metadata.fileSize > 1024 * 1024 
            ? `${(metadata.fileSize / (1024 * 1024)).toFixed(1)}MB`
            : metadata.fileSize > 1024
                ? `${(metadata.fileSize / 1024).toFixed(1)}KB`
                : `${metadata.fileSize}B`;
                
        // Log actual file size for debugging
        if (this.debugEnabled) {
            console.log(`[MetadataVisualizer] File size for node ${nodeId}: ${metadata.fileSize} bytes (${fileSizeFormatted})`);
        }

        // Only log detailed metadata at trace level (effectively disabling it)
        if (debugState.isDataDebugEnabled()) {
            this.logger.debug(`Creating metadata label #${this.labelUpdateCount}:`, {
                nodeId,
                metadata: {
                    name: metadata.name,
                    fileSize: fileSizeFormatted,
                    nodeSize: metadata.nodeSize
                }
            });
        }

        // Use metadata name, ensuring we show a properly formatted name
        const displayName = metadata.name || nodeId.toString();

        // Create text labels using UnifiedTextRenderer
        // First, find the node's actual position
        let nodePosition = new Vector3(0, 0, 0);
        
        // Get the actual node position from some source in the scene
        // This solves the labels "dropping in from above" issue
        if (debugState.isDataDebugEnabled()) {
            this.logger.debug(`Searching for node position for ${nodeId}`);
        }
        
        // We'll set the group's position initially to help with initialization
        group.position.copy(nodePosition);
        
        // Enhanced label text that displays the filename prominently
        const labelTexts = [
            `${displayName}`,  // Main label (filename)
            `${fileSizeFormatted}`,  // File size
            `${metadata.hyperlinkCount || 0} links`  // Link count
        ];

        const yOffsets = [0.05, 0.03, 0.01]; // Drastically reduced Y positions to keep labels almost at node position

        labelTexts.forEach(async (text, index) => {
            const position = new Vector3(nodePosition.x, nodePosition.y + yOffsets[index], nodePosition.z);
            const labelId = `${nodeId}-label-${index}`;
            
            try {
                this.textRenderer.updateLabel(
                    labelId,
                    text,
                    position,
                    new Color(this.settings.visualization.labels.textColor)
                );
                // Only log when specific data debugging is enabled
                if (debugState.isDataDebugEnabled()) {
                    this.logger.debug(`Created label ${index+1}/3 for node ${nodeId}`, {
                        labelId
                    });
                }
            } catch (error) {
                this.logger.error(`Failed to create label ${index+1}/3 for node ${nodeId}`, {
                    error: error instanceof Error ? error.message : String(error),
                    labelId,
                    text
                });
            }
        });

        this.metadataGroups.set(nodeId, group);
        
        // Add call to update position immediately to prevent "dropping in" effect
        this.updateMetadataPosition(nodeId, nodePosition);
        
        return group;
    }

    private setGroupLayer(group: Object3D, enabled: boolean): void {
        this.logger.debug(`Setting layer mode: ${enabled ? 'XR' : 'Desktop'}`);
        
        if (enabled) {
            group.traverse(child => {
                child.layers.disable(0);
                child.layers.enable(1);
            });
            group.layers.disable(0);
            group.layers.enable(1);
        } else {
            group.traverse(child => {
                child.layers.enable(0);
                child.layers.enable(1);
            });
            group.layers.enable(0);
            group.layers.enable(1);
        }
    }

    /**
     * Update visibility threshold for labels
     */
    public setXRMode(enabled: boolean): void {
        this.logger.info(`Switching to ${enabled ? 'XR' : 'Desktop'} mode`);
        this.textRenderer.setXRMode(enabled);
        this.setGroupLayer(this.labelGroup, enabled);
        // Text renderer handles its own XR mode
    }

    public updateMetadataPosition(nodeId: string, position: Vector3): void {
        /**
         * CRITICAL NODE ID BINDING: Position Updates
         * 
         * This method is called from VisualizationController to update the position
         * of a metadata label when node positions change via the WebSocket.
         * 
         * The nodeId parameter MUST match:
         * 1. The ID used in createMetadataLabel() to create the label
         * 2. The IDs coming from the binary WebSocket protocol
         * 
         * If labels don't move with their nodes, it's likely due to a mismatch
         * between the ID used here and the ID used in the binary protocol.
         */
        const group = this.metadataGroups.get(nodeId);
        if (group) {
            group.position.copy(position);
            
            // Update text positions
            const labelPositions = [0.05, 0.03, 0.01]; // Drastically reduced offsets to match createMetadataLabel
            labelPositions.forEach((yOffset, index) => {
                const labelId = `${nodeId}-label-${index}`;
                // Create relative position to the node with y-offset
                const relativePosition = new Vector3(0, yOffset, 0);
                const labelPosition = position.clone().add(relativePosition);
                this.textRenderer.updateLabel(labelId, '', labelPosition); // Text content remains unchanged
                
                // Only log position updates when specific data debugging is enabled
                if (index === 0 && debugState.isDataDebugEnabled()) {
                    this.logger.debug(`Updating label position for ${nodeId}`, {
                        labelId });
                }
                
                // Only show debug helpers when debug is enabled
                if (debugState.isEnabled()) {
                    const debugId = `${labelId}-debug`;
                    let debugSphere = this.debugHelpers.get(debugId) as Mesh | undefined;
                    if (!debugSphere) {
                        const geometry = new SphereGeometry(0.1);
                        const material = new MeshBasicMaterial({ color: 0xff0000 });
                        debugSphere = new Mesh(geometry, material) as Mesh;
                        this.labelGroup.add(debugSphere);
                        this.debugHelpers.set(debugId, debugSphere);
                    }
                    debugSphere.position.copy(labelPosition);
                    debugSphere.visible = true;
                }
            });
        }
    }

    /**
     * Updates the visibility threshold for metadata labels
     */
    public updateVisibilityThreshold(threshold: number): void {
        this.visibilityThreshold = threshold;
        this.logger.info('Updated visibility threshold:', { 
            threshold, 
            labelsCount: this.metadataGroups.size 
        });
    }

    public removeMetadata(nodeId: string): void {
        const group = this.metadataGroups.get(nodeId);
        if (group) {
            this.labelGroup.remove(group);
            this.metadataGroups.delete(nodeId);
            
            // Remove text labels
            [0, 1, 2].forEach(index => {
                const labelId = `${nodeId}-label-${index}`;
                this.textRenderer.removeLabel(labelId);
                const debugId = `${labelId}-debug`;
                
                if (debugState.isEnabled()) {
                    // Remove debug helpers
                    const debugHelper = this.debugHelpers.get(debugId);
                    if (debugHelper) this.labelGroup.remove(debugHelper);
                }
                if (this.debugHelpers.has(debugId)) this.debugHelpers.delete(debugId);
            });
        }
    }
    
    /**
     * Update all metadata labels - called once per frame
     */
    public update(_camera: PerspectiveCamera): void {
        // Very rarely log how many labels we're tracking
        if (Math.random() < 0.0001 && this.metadataGroups.size > 0) {
            // Only log if data debugging is specifically enabled
            if (debugState.isDataDebugEnabled()) {
                this.logger.debug('Metadata update stats:', {
                    labelsCount: this.metadataGroups.size,
                    enabled: this.settings.visualization.labels.enableLabels,
                    rendererActive: this.textRenderer !== null,
                    visibilityThreshold: this.visibilityThreshold
                });
            }
        }
        // The text renderer handles label positions and visibility
    }

    public dispose(): void {
        this.metadataGroups.forEach(group => {
            if (group.userData.nodeId) {
                try {
                    this.removeMetadata(group.userData.nodeId);
                } catch (e) {
                    this.logger.error(`Error removing metadata for node ${group.userData.nodeId}`, {
                        error: e instanceof Error ? e.message : String(e)
                    });
                }
            }
        });
        this.metadataGroups.clear();
        // Don't dispose the text renderer itself, as we'll reuse it
        // this.textRenderer.dispose();
        if (this.labelGroup.parent) {
            // Clean up debug helpers
            this.debugHelpers.forEach(helper => {
                this.labelGroup.remove(helper);
            });
            this.debugHelpers.clear();
            // Don't remove the label group from the scene, just clear it
            // this.labelGroup.parent.remove(this.labelGroup);
        }
        
        this.logger.info('Cleared all metadata visualizations');
    }
    
    /**
     * Clear all label content without fully disposing
     */
    public clearAllLabels(): void {
        // Store node IDs before clearing
        const nodeIds = Array.from(this.metadataGroups.keys());
        
        // Debounce clear operations to prevent excessive clearing
        const now = performance.now();
        if (now - this.lastClearTime < 1000) {
            // If a clear was performed in the last second, just log and return
            this.logger.debug(`Skipping redundant label clear operation (too soon after last clear)`);
            return;
        }
        this.lastClearTime = now;
        
        // Remove all labels
        nodeIds.forEach(nodeId => {
            this.removeMetadata(nodeId);
        });
        
        this.metadataGroups.clear();
        this.debugHelpers.clear();
        
        this.logger.info(`Cleared ${nodeIds.length} metadata labels`);
    }
}
