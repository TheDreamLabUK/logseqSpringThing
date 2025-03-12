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
    public processNodes(nodes: any[]): { duplicateLabels: Map<string, string[]> } {
        const duplicateLabels = new Map<string, string[]>();
        
        // Clear the label tracking for this batch to detect duplicates
        this.labelToNodeIds.clear();
        
        // Process each node
        nodes.forEach(node => {
            if (!this.isValidNumericId(node.id)) {
                logger.warn(`Skipping node with invalid ID format: ${node.id}`);
                return;
            }
            
            // Determine the best label using a simplified approach
            const label = this.determineBestLabel(node);
            
            // Store the mapping
            if (label && label !== node.id) {
                this.numericIdToLabel.set(node.id, label);
                this.trackLabelUsage(node.id, label);
            }
        });
        
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
        if (!label || label === nodeId || this.NODE_ID_REGEX.test(label)) {
            return; // Don't track numeric IDs or empty labels
        }
        
        if (!this.labelToNodeIds.has(label)) {
            this.labelToNodeIds.set(label, [nodeId]);
        } else {
            const existingNodes = this.labelToNodeIds.get(label)!;
            if (!existingNodes.includes(nodeId)) {
                existingNodes.push(nodeId);
            }
        }
    }
    
    /**
     * Determines the best label to use for a node using a simplified priority order
     */
    private determineBestLabel(node: any): string {
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
        
        // Last resort: use the numeric ID itself
        return node.id;
    }
    
    /**
     * Get the label for a numeric node ID
     */
    public getLabel(numericId: string): string {
        return this.numericIdToLabel.get(numericId) || numericId;
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