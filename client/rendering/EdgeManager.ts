import {
    BufferGeometry,
    BufferAttribute,
    LineBasicMaterial,
    Mesh,
    Scene,
    Vector3,
    Material
} from 'three';
import { createLogger } from '../core/logger';
import { Edge } from '../core/types';
import { Settings } from '../types/settings';
import { NodeInstanceManager } from './node/instance/NodeInstanceManager';

const logger = createLogger('EdgeManager');

export class EdgeManager {
    private scene: Scene;
    private settings: Settings;
    private edges: Map<string, Mesh> = new Map(); // Store edges by ID
    private nodeInstanceManager: NodeInstanceManager; // Reference to NodeInstanceManager

    constructor(scene: Scene, settings: Settings, nodeInstanceManager: NodeInstanceManager) {
        this.scene = scene;
        this.settings = settings;
        this.nodeInstanceManager = nodeInstanceManager;
        logger.info('EdgeManager initialized');
        
        // Add constructor validation
        if (!this.scene) {
            logger.error("Scene is null or undefined in EdgeManager constructor");
        }
        if (!this.settings) {
            logger.error("Settings are null or undefined in EdgeManager constructor");
        }
        if (!this.nodeInstanceManager) {
            logger.error("NodeInstanceManager is null or undefined in EdgeManager constructor");
        }
    }

    /**
     * Validates a Vector3 to ensure it has valid, finite values
     */
    private validateVector3(vec: Vector3): boolean {
        const MAX_VALUE = 1000;
        return isFinite(vec.x) && isFinite(vec.y) && isFinite(vec.z) &&
             !isNaN(vec.x) && !isNaN(vec.y) && !isNaN(vec.z) &&
             Math.abs(vec.x) < MAX_VALUE && Math.abs(vec.y) < MAX_VALUE && Math.abs(vec.z) < MAX_VALUE;
    }

    public updateEdges(edges: Edge[]): void {
        logger.info(`Updating ${edges.length} edges, current edge count: ${this.edges.size}`);

        const newEdges: Edge[] = [];
        const existingEdges: Edge[] = [];

        const currentEdgeIds = new Set(edges.map(edge => this.createEdgeId(edge.source, edge.target)));
        for (const edge of edges) {
            // Use numeric IDs for edge identification
            const edgeId = this.createEdgeId(edge.source, edge.target);

            logger.debug(`Checking edge: ${edgeId}`);

            if (this.edges.has(edgeId)) {
                existingEdges.push(edge);
            } else {
                newEdges.push(edge);
            }
        }

        logger.debug(`Found ${newEdges.length} new edges and ${existingEdges.length} existing edges`);

        // Add new edges
        for (const edge of newEdges) {
            logger.debug(`Creating edge: ${edge.source}-${edge.target}`);
            this.createEdge(edge);
        }

        // Update existing edges (positions might have changed)
        for (const edge of existingEdges) {
            logger.debug(`Updating edge: ${edge.source}-${edge.target}`);
            this.updateEdge(edge);
        }

        // Remove old edges (not in the new set)
        for (const edgeId of this.edges.keys()) {
            if (!currentEdgeIds.has(edgeId)) {
                this.removeEdge(edgeId);
            }
        }
    }
  
    private createEdge(edge: Edge): void {
        const edgeId = this.createEdgeId(edge.source, edge.target);

        // Validate source and target IDs - they should be numeric strings
        if (!this.validateNodeId(edge.source) || !this.validateNodeId(edge.target)) {
            logger.warn(`Skipping edge creation with invalid node IDs: source=${edge.source}, target=${edge.target}`);
            return;
        }
        // Get node positions from NodeInstanceManager using numeric IDs
        const sourcePos = this.nodeInstanceManager.getNodePosition(edge.source);
        const targetPos = this.nodeInstanceManager.getNodePosition(edge.target);
        
        logger.debug(`Creating edge ${edgeId}`, { 
            source: edge.source,
            target: edge.target,
            sourcePos: sourcePos ? [sourcePos.x, sourcePos.y, sourcePos.z] : null,
            targetPos: targetPos ? [targetPos.x, targetPos.y, targetPos.z] : null
        });

        if (!sourcePos || !targetPos) {
            logger.warn(`Skipping edge creation for ${edgeId} due to missing node positions. Source exists: ${!!sourcePos}, Target exists: ${!!targetPos}`);
            return;
        }

        const isSourceValid = this.validateVector3(sourcePos);
        const isTargetValid = this.validateVector3(targetPos);

        if (!isSourceValid || !isTargetValid) {
            logger.warn(`Skipping edge creation for ${edgeId} due to invalid node positions. Source valid: ${isSourceValid}, Target valid: ${isTargetValid}`);
            if (!isSourceValid) {
                logger.warn(`Invalid source position: [${sourcePos.x}, ${sourcePos.y}, ${sourcePos.z}]`);
            }
            if (!isTargetValid) {
                logger.warn(`Invalid target position: [${targetPos.x}, ${targetPos.y}, ${targetPos.z}]`);
            }
            return;
        }

        // Create a simple line geometry
        const geometry = new BufferGeometry();
        const positions = new Float32Array([
            sourcePos.x, sourcePos.y, sourcePos.z,
            targetPos.x, targetPos.y, targetPos.z
        ]);
        
        geometry.setAttribute('position', new BufferAttribute(positions, 3));
        
        // Create LineBasicMaterial with higher opacity for better visibility
        const material = new LineBasicMaterial({
            color: this.settings.visualization.edges.color || "#888888", 
            transparent: true, 
            // Use settings opacity directly instead of multiplying
            opacity: this.settings.visualization.edges.opacity || 0.8
        });

        // Use Mesh with line geometry for rendering
        const line = new Mesh(geometry, material);
        line.renderOrder = 5; // Increased to render on top of nodes

        // Store the edge ID in userData for identification
        line.userData = { edgeId };
        
        // Enable both layers by default for desktop mode
        line.layers.enable(0);
        line.layers.enable(1);
        
        this.edges.set(edgeId, line);
        
        // Add to scene and check
        this.scene.add(line);
        
        // Verify the edge was added to the scene
        logger.debug(`Edge created: ${edgeId}, visible: ${line.visible}, layers: ${line.layers.mask}`);
        
        // Log the material properties
        const mat = line.material as LineBasicMaterial;
        logger.debug(`Edge material: color=${mat.color.toString()}, opacity=${mat.opacity}, transparent=${mat.transparent}`);
    }

    /**
     * Creates a consistent edge ID by sorting source and target IDs
     * This ensures the same ID regardless of edge direction
     */
    private createEdgeId(source: string, target: string): string {
        return [source, target].sort().join('_');
    }

    private validateNodeId(id: string): boolean {
        return id !== undefined && id !== null && /^\d+$/.test(id);
    }

    private updateEdge(edge: Edge): void {
        const edgeId = this.createEdgeId(edge.source, edge.target);
        const line = this.edges.get(edgeId);
        
        if (!line) {
            this.createEdge(edge);
            return;
        }

        // Get updated node positions
        const sourcePos = this.nodeInstanceManager.getNodePosition(edge.source);
        const targetPos = this.nodeInstanceManager.getNodePosition(edge.target);

        if (!sourcePos || !targetPos) {
            logger.warn(`Cannot update edge ${edgeId}: missing node positions`);
            return;
        }

        // Validate positions
        const isSourceValid = this.validateVector3(sourcePos);
        const isTargetValid = this.validateVector3(targetPos);
        
        if (!isSourceValid || !isTargetValid) {
            logger.warn(`Skipping edge update for ${edgeId} due to invalid node positions. ` + 
                        `Source valid: ${isSourceValid}, Target valid: ${isTargetValid}`);
            if (!isSourceValid) logger.warn(`Invalid source position: [${sourcePos.x}, ${sourcePos.y}, ${sourcePos.z}]`);
            if (!isTargetValid) logger.warn(`Invalid target position: [${targetPos.x}, ${targetPos.y}, ${targetPos.z}]`);
            return;
        }

        // Update the geometry with new positions
        const positions = new Float32Array([
            sourcePos.x, sourcePos.y, sourcePos.z,
            targetPos.x, targetPos.y, targetPos.z
        ]);
        
        line.geometry.dispose();
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(positions, 3)); 
        
        line.geometry = geometry;
        
        logger.debug(`Edge updated: ${edgeId}`);
    }

    private removeEdge(edgeId: string): void {
        const edge = this.edges.get(edgeId);
        if (edge) {
            logger.debug(`Removing edge: ${edgeId}`);
            this.scene.remove(edge);
            edge.geometry.dispose();
            if (Array.isArray(edge.material)) {
                logger.debug(`Disposing ${edge.material.length} materials for edge ${edgeId}`);
                edge.material.forEach((m: Material) => m.dispose());
            } else {
                logger.debug(`Disposing material for edge ${edgeId}`);
                edge.material.dispose();
            }
            this.edges.delete(edgeId);
        } else {
            logger.warn(`Attempted to remove non-existent edge: ${edgeId}`);
        }
    }

    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        // Update edge appearance based on new settings
        this.edges.forEach((edge) => {
            if (edge.material instanceof LineBasicMaterial) {
                edge.material.color.set(this.settings.visualization.edges.color);
                edge.material.opacity = this.settings.visualization.edges.opacity;
                edge.material.needsUpdate = true;
            }
        });
    }

    /**
     * Set XR mode for edge rendering
     */
    public setXRMode(enabled: boolean): void {
        // Set appropriate layer visibility for all edges
        logger.info(`Setting XR mode for ${this.edges.size} edges: ${enabled ? 'enabled' : 'disabled'}`);
        this.edges.forEach(edge => {
            if (enabled) {
                // In XR mode, only show on layer 1
                edge.layers.disable(0);
                edge.layers.enable(1);
            } else {
                // In desktop mode, show on both layers
                edge.layers.enable(0);
                edge.layers.enable(1);
            }
        });
    }

    public update(): void {
        // The edge update is now handled during updateEdges
        // This method is kept for compatibility with the rendering loop
    }

    /**
     * Clean up resources
     */
    public dispose(): void {
        // Dispose of all geometries and materials
        this.edges.forEach(edge => {
            this.scene.remove(edge);
            edge.geometry.dispose();
            if (Array.isArray(edge.material)) {
                edge.material.forEach((m: Material) => m.dispose());
            } else {
                edge.material.dispose();
            }
        });
        this.edges.clear();
    }
}
