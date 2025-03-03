import {
    BufferGeometry,
    BufferAttribute,
    LineBasicMaterial,
    Mesh,
    Scene,
    Vector3,
    Material,
    Color,
    Matrix4,
    Object3D
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
    private tempMatrix = new Matrix4();

    constructor(scene: Scene, settings: Settings, nodeInstanceManager: NodeInstanceManager) {
        this.scene = scene;
        this.settings = settings;
        this.nodeInstanceManager = nodeInstanceManager;
        logger.info('EdgeManager initialized');
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
        logger.info(`Updating ${edges.length} edges`);

        const newEdges: Edge[] = [];
        const existingEdges: Edge[] = [];

        for (const edge of edges) {
            // Use numeric IDs for edge identification
            const edgeId = `${edge.source}-${edge.target}`;

            if (this.edges.has(edgeId)) {
                existingEdges.push(edge);
            } else {
                newEdges.push(edge);
            }
        }

        // Add new edges
        for (const edge of newEdges) {
            this.createEdge(edge);
        }

        // Update existing edges (positions might have changed)
        for (const edge of existingEdges) {
            this.updateEdge(edge);
        }

        // Remove old edges (not in the new set)
        const currentEdgeIds = new Set(edges.map(edge => `${edge.source}-${edge.target}`));
        for (const edgeId of this.edges.keys()) {
            if (!currentEdgeIds.has(edgeId)) {
                this.removeEdge(edgeId);
            }
        }
    }
  
    private createEdge(edge: Edge): void {
        const edgeId = `${edge.source}-${edge.target}`;

        // Get node positions from NodeInstanceManager using numeric IDs
        const sourcePos = this.nodeInstanceManager.getNodePosition(edge.source);
        const targetPos = this.nodeInstanceManager.getNodePosition(edge.target);

        if (!sourcePos || !targetPos) {
            // Handle cases where node positions might not be available yet
            logger.warn(`Skipping edge creation for ${edgeId} due to missing node positions`);
            return;
        }

        // Validate positions
        if (!this.validateVector3(sourcePos) || !this.validateVector3(targetPos)) {
            logger.warn(`Skipping edge creation for ${edgeId} due to invalid node positions`);
            return;
        }

        // Create a simple line geometry
        const geometry = new BufferGeometry();
        const positions = new Float32Array([
            sourcePos.x, sourcePos.y, sourcePos.z,
            targetPos.x, targetPos.y, targetPos.z
        ]);
        geometry.setAttribute('position', new BufferAttribute(positions, 3));
        
        const material = new LineBasicMaterial({
            color: this.settings.visualization.edges.color,
            transparent: true,
            opacity: this.settings.visualization.edges.opacity
        });

        // Create a Mesh for line rendering
        // Note: We're using Mesh but with a line-like geometry
        const line = new Mesh(geometry, material);
        line.renderOrder = 1; // Ensure lines render on top of nodes

        // Store the edge ID in userData for identification
        line.userData = { edgeId };
        
        // Enable both layers by default for desktop mode
        line.layers.enable(0);
        line.layers.enable(1);
        
        this.edges.set(edgeId, line);
        this.scene.add(line);
    }

    private updateEdge(edge: Edge): void {
        const edgeId = `${edge.source}-${edge.target}`;
        const line = this.edges.get(edgeId);
        
        if (!line) {
            this.createEdge(edge);
            return;
        }

        // Get updated node positions
        const sourcePos = this.nodeInstanceManager.getNodePosition(edge.source);
        const targetPos = this.nodeInstanceManager.getNodePosition(edge.target);

        if (!sourcePos || !targetPos) {
            return;
        }

        // Validate positions
        if (!this.validateVector3(sourcePos) || !this.validateVector3(targetPos)) {
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
    }

    private removeEdge(edgeId: string): void {
        const edge = this.edges.get(edgeId);
        if (edge) {
            this.scene.remove(edge);
            edge.geometry.dispose();
            if (Array.isArray(edge.material)) {
                edge.material.forEach((m: Material) => m.dispose());
            } else {
                edge.material.dispose();
            }
            this.edges.delete(edgeId);
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

    public update(deltaTime: number): void {
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
