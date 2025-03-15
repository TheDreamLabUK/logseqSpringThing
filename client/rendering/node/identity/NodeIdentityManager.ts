import { createLogger, createDataMetadata } from '../../../core/logger';

const logger = createLogger('NodeIdentityManager');

/**
 * NodeIdentityManager provides a single source of truth for node identity resolution.
 * 
 * It simplifies the relationship between:
 * - Server-generated numeric node IDs (u16 indices)
 * - Human-readable labels (filenames without .md extension)
 */
export class NodeIdentityManager {
    private static instance: NodeIdentityManager;
    
    // Maps numeric node ID to user-friendly label
    private numericIdToLabel = new Map<string, string>();
    
    // Maps labels to the numeric IDs that use them (for duplicate detection)
    private labelToNodeIds = new Map<string, string[]>();
    
    // Regular expression to validate numeric IDs
    private readonly NODE_ID_REGEX = /^\d+$/;
    
    // Set default label to ensure it's never empty or duplicated
    private readonly DEFAULT_LABEL_PREFIX = 'Node_';
    
    private constructor() {
        logger.info('NodeIdentityManager initialized');
    }
    
    /**
     * Get the singleton instance of NodeIdentityManager
     */
    public static getInstance(): NodeIdentityManager {
        if (!NodeIdentityManager.instance) {
            NodeIdentityManager.instance = new NodeIdentityManager();
        }
        return NodeIdentityManager.instance;
    }
    
    /**
     * Check if a string is a valid numeric node ID
     */
    public isValidNumericId(id: string): boolean {
        return typeof id === 'string' && this.NODE_ID_REGEX.test(id);
    }
    
    /**
     * Process a batch of nodes to establish identity mappings and detect duplicates
     */
    public processNodes(nodes: any[], forceReset: boolean = false): { duplicateLabels: Map<string, string[]> } {
        const duplicateLabels = new Map<string, string[]>();
        
        // If force reset is requested, completely reset our mappings
        if (forceReset) {
            logger.info('Forced reset of all node identity mappings');
            this.reset();
        }
        
        if (nodes.length === 0) {
            return { duplicateLabels };
        }
        
        // Log first few nodes for debugging
        logger.info(`Processing ${nodes.length} nodes. First few node IDs:`, 
            createDataMetadata({
                nodeIds: nodes.slice(0, 5).map(n => n.id)
            }));
        
        // Track processed nodes in this batch
        const processedNodeIds = new Set<string>();
        
        // Process each node
        nodes.forEach(node => {
            if (!this.isValidNumericId(node.id)) {
                logger.warn(`Skipping node with invalid ID format: ${node.id}`);
                return;
            }
            
            processedNodeIds.add(node.id);
            
            // Get any existing label for this node
            const existingLabel = this.numericIdToLabel.get(node.id);
            
            if (existingLabel) {
                // If we have an existing label and it's different from the node ID, use it
                // But only if it doesn't look like our default or a numeric ID
                if (existingLabel !== node.id && 
                    !this.NODE_ID_REGEX.test(existingLabel) && 
                    !existingLabel.startsWith(this.DEFAULT_LABEL_PREFIX)) {
                    
                    // Keep using the existing label if it's valid
                    this.trackLabelUsage(node.id, existingLabel);
                    return;
                }
            }
            
            // Determine a new best label if we don't have a valid one
            const newLabel = this.determineBestLabel(node);
            if (newLabel && newLabel !== node.id) {
                this.numericIdToLabel.set(node.id, newLabel);
                this.trackLabelUsage(node.id, newLabel);
            }
        });
        
        // Rebuild the labelToNodeIds map based on all existing mappings in numericIdToLabel
        // but keeping any entries for nodes not in this batch
        const newLabelToNodeIds = new Map<string, string[]>();
        
        // First, copy over existing entries for nodes not in this batch
        this.labelToNodeIds.forEach((nodeIds, label) => {
            const filteredNodeIds = nodeIds.filter(id => !processedNodeIds.has(id));
            if (filteredNodeIds.length > 0) {
                newLabelToNodeIds.set(label, filteredNodeIds);
            }
        });
        
        // Then add the entries from this batch
        this.numericIdToLabel.forEach((label, nodeId) => {
            if (!label || label === nodeId || this.NODE_ID_REGEX.test(label)) {
                return; // Skip numeric IDs or empty labels
            }
            
            if (!newLabelToNodeIds.has(label)) {
                newLabelToNodeIds.set(label, [nodeId]);
            } else {
                newLabelToNodeIds.get(label)!.push(nodeId);
            }
        });
        
        // Update the main map
        this.labelToNodeIds = newLabelToNodeIds;
        
        // Find and log duplicates
        this.labelToNodeIds.forEach((nodeIds, label) => {
            if (nodeIds.length > 1) {
                // Only track meaningful labels, not numeric IDs or generic names
                if (!this.NODE_ID_REGEX.test(label) && 
                    !['undefined', 'Unknown', ''].includes(label)) {
                    duplicateLabels.set(label, [...nodeIds]);
                    logger.warn(`DUPLICATE LABEL: "${label}" is used by ${nodeIds.length} nodes: ${nodeIds.join(', ')}`, 
                        createDataMetadata({ label, nodeCount: nodeIds.length, nodeIds }));
                }
            }
        });
        
        return { duplicateLabels };
    }
    
    /**
     * Track which labels are used to detect duplicates
     */
    private trackLabelUsage(nodeId: string, label: string): void {
        if (!nodeId || !label || label === nodeId || this.NODE_ID_REGEX.test(label)) {
            return; // Don't track numeric IDs or empty labels
        }
        
        // Ensure we're not trying to add an invalid nodeId
        if (!this.isValidNumericId(nodeId)) {
            logger.warn(`Attempted to track invalid node ID: ${nodeId} for label: ${label}`);
            return;
        }
        
        if (!this.labelToNodeIds.has(label)) {
            this.labelToNodeIds.set(label, [nodeId]);
        } else {
            this.labelToNodeIds.get(label)!.push(nodeId);
        }
    }
    
    /**
     * Determines the best label to use for a node using a simplified priority order
     */
    private determineBestLabel(node: any, fallbackToDefault: boolean = true): string {
        let bestLabel = '';
        // First try metadata.name as the canonical source of the filename without extension
        if (node.data?.metadata?.name && typeof node.data.metadata.name === 'string' && 
            node.data.metadata.name !== 'undefined') {
            return node.data.metadata.name;
        }
        
        // Next try explicitly provided label
        if (typeof node.label === 'string' && node.label && 
            node.label !== 'undefined' && !this.NODE_ID_REGEX.test(node.label)) {
            return node.label;
        }
        
        // Next try metadataId (which is typically the filename)
        if (typeof node.metadataId === 'string' && node.metadataId && 
            node.metadataId !== 'undefined' && !this.NODE_ID_REGEX.test(node.metadataId)) {
            return node.metadataId;
        }
        
        // If existing approaches failed and we should use a default
        if (fallbackToDefault) {
            // Always ensure uniqueness by creating a unique label using numeric ID as suffix
            bestLabel = `${this.DEFAULT_LABEL_PREFIX}${node.id}`;
            
            // Log this default label creation as it should only happen rarely
            if (Math.random() < 0.1) {
                logger.debug(`Created default label for node ${node.id}: ${bestLabel}`);
            }
            
            return bestLabel;
        }
        
        // Absolute last resort: use the numeric ID itself
        logger.debug(`Failed to determine label for node ${node.id}, using ID as fallback`);
        return node.id;
    }
    
    /**
     * Force a label for a node, useful when a specific label must be assigned
     */
    public forceNodeLabel(nodeId: string, label: string): void {
        if (this.isValidNumericId(nodeId)) {
            this.numericIdToLabel.set(nodeId, label);
            this.trackLabelUsage(nodeId, label);
        }
    }
    
    /**
     * Get the label for a numeric node ID
     */
    public getLabel(numericId: string): string {
        if (!this.isValidNumericId(numericId)) {
            return numericId; // If not a valid ID, just return it
        }
        
        const label = this.numericIdToLabel.get(numericId);
        
        // If we have a label and it's not a numeric ID or default label
        if (label && !this.NODE_ID_REGEX.test(label) && !label.startsWith(this.DEFAULT_LABEL_PREFIX)) {
            return label;
        }
        
        // If we don't have a label, or it's just a numeric ID or default label,
        // create a guaranteed unique label based on the numeric ID
        const newLabel = `${this.DEFAULT_LABEL_PREFIX}${numericId}`;
        
        // Save this for future use
        this.numericIdToLabel.set(numericId, newLabel);
        this.trackLabelUsage(numericId, newLabel);
        
        logger.debug(`Generated new label for node ${numericId}: ${newLabel}`);
        
        return newLabel;
    }
    
    /**
     * Get all numeric node IDs that have been processed
     * @returns Array of numeric node IDs
     */
    public getAllNodeIds(): string[] {
        return Array.from(this.numericIdToLabel.keys());
    }
    
    /**
     * Set the label for a numeric node ID
     */
    public setLabel(numericId: string, label: string): void {
        if (!numericId || !label) return;
        this.numericIdToLabel.set(numericId, label);
        this.trackLabelUsage(numericId, label);
    }
    
    /**
     * Get all nodes that use a specific label
     */
    public getNodesWithLabel(label: string): string[] {
        return this.labelToNodeIds.get(label) || [];
    }
    
    /**
     * Check if a label is used by multiple nodes
     */
    public isDuplicateLabel(label: string): boolean {
        const nodes = this.labelToNodeIds.get(label);
        return !!nodes && nodes.length > 1;
    }
    
    /**
     * Get all duplicate labels
     */
    public getDuplicateLabels(): Map<string, string[]> {
        const duplicates = new Map<string, string[]>();
        this.labelToNodeIds.forEach((nodeIds, label) => {
            if (nodeIds.length > 1 && !this.NODE_ID_REGEX.test(label) && label) {
                duplicates.set(label, [...nodeIds]);
            }
        });
        return duplicates;
    }
    
    /**
     * Reset all mappings
     */
    public reset(): void {
        this.numericIdToLabel.clear();
        this.labelToNodeIds.clear();
        logger.info('All node identity mappings reset');
    }
    
    /**
     * Dispose of the manager and clear all mappings
     */
    public dispose(): void {
        this.reset();
        NodeIdentityManager.instance = null!;
    }
}