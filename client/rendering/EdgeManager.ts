import {
    BufferGeometry,
    BufferAttribute,
    LineBasicMaterial,
    Mesh,
    Vector3,
    Scene,
    Group,
    Object3D,
    Material,
    Color
} from 'three';
import { createLogger } from '../core/logger';
import { Edge } from '../core/types';
import { Settings } from '../types/settings';
import { NodeInstanceManager } from './node/instance/NodeInstanceManager';
import { SettingsStore } from '../state/SettingsStore';
import { debugState } from '../core/debugState';

const logger = createLogger('EdgeManager');

export class EdgeManager {
    private scene: Scene;
    private edges: Map<string, Mesh> = new Map();
    private edgeGroup: Group;
    private nodeManager: NodeInstanceManager;
    private edgeData: Map<string, Edge> = new Map();
    private sourceTargetCache: Map<string, string> = new Map();
    private settings: Settings;
    private settingsStore: SettingsStore;
    private updateFrameCount = 0;
    private readonly UPDATE_FREQUENCY = 2; // Update every other frame
    private readonly MAX_EDGE_LENGTH = 20.0; // Increased maximum edge length

    constructor(scene: Scene, settings: Settings, nodeManager: NodeInstanceManager) {
        this.scene = scene;
        this.nodeManager = nodeManager;
        this.settings = settings;
        this.settingsStore = SettingsStore.getInstance();
        this.edgeGroup = new Group();
        
        // Enable both layers by default for desktop mode
        this.edgeGroup.layers.enable(0);
        this.edgeGroup.layers.enable(1);
        
        scene.add(this.edgeGroup);

        // Subscribe to settings changes
        this.settingsStore.subscribe('visualization.edges', (_: string, settings: any) => {
            if (settings && typeof settings === 'object') {
                this.settings = {
                    ...this.settings,
                    visualization: {
                        ...this.settings.visualization,
                        edges: settings
                    }
                };
                this.handleSettingsUpdate(this.settings);
            }
        });
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

    /**
     * Creates an optimized line geometry for an edge between two points
     */
    private createLineGeometry(source: Vector3, target: Vector3): BufferGeometry {
        const geometry = new BufferGeometry();
        
        // Apply maximum edge length limit to prevent explosion
        const distance = source.distanceTo(target);
        const finalTarget = new Vector3().copy(target);
        
        if (distance > this.MAX_EDGE_LENGTH) {
            // Create limited-length vector in same direction
            const direction = new Vector3().subVectors(target, source).normalize();
            // Direction check to prevent issues with zero vectors
            if (isNaN(direction.x) || isNaN(direction.y) || isNaN(direction.z)) return geometry;
            finalTarget.copy(source).add(direction.multiplyScalar(this.MAX_EDGE_LENGTH));
        }
        
        // Line geometry only needs the start and end positions
        const positions = new Float32Array([
            source.x, source.y, source.z,
            finalTarget.x, finalTarget.y, finalTarget.z
        ]);

        geometry.setAttribute('position', new BufferAttribute(positions, 3));
        return geometry;
    }

    /**
     * Creates an optimized material for edge rendering
     */
    private createEdgeMaterial(): Material {
        const color = new Color(this.settings.visualization.edges.color || 0x4080ff);
        const opacity = this.settings.visualization.edges.opacity || 0.7;
        
        return new LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity, 
            depthWrite: false
        });
    }

    /**
     * Updates all edges with new data from the graph
     */
    public updateEdges(edges: Edge[]): void {
        // Debug logging - only if node debug is enabled
        if (debugState.isNodeDebugEnabled()) {
            logger.debug(`Updating ${edges.length} edges`);
        }
        
        // Clear existing edges
        this.clearEdges();
        
        // Clear maps
        this.edgeData.clear();
        this.edges.clear();
        this.sourceTargetCache.clear();
        
        // Track counts for debugging
        let edgesCreated = 0;
        let edgesSkipped = 0;

        // Create new edges
        edges.forEach(edge => {
            // Cache mapping between source and target IDs
            this.sourceTargetCache.set(edge.id || `${edge.source}_${edge.target}`, `${edge.source}:${edge.target}`);
            
            if (!edge.sourcePosition || !edge.targetPosition) { edgesSkipped++; return; }

            // Clamp positions to reasonable values
            const source = new Vector3(
                Math.min(100, Math.max(-100, edge.sourcePosition.x)),
                Math.min(100, Math.max(-100, edge.sourcePosition.y)),
                Math.min(100, Math.max(-100, edge.sourcePosition.z))
            );
            
            const target = new Vector3(
                Math.min(100, Math.max(-100, edge.targetPosition.x)),
                Math.min(100, Math.max(-100, edge.targetPosition.y)),
                Math.min(100, Math.max(-100, edge.targetPosition.z))
            );

            // Skip edge creation if source and target are too close
            const distance = source.distanceTo(target);
            if (distance < 0.001) {
                if (debugState.isNodeDebugEnabled()) {
                    logger.debug(`Skipping edge ${edge.id} - nodes too close: distance=${distance.toFixed(6)}`);
                }
                edgesSkipped++;
                return;
            } else if (distance < 0.1) {
                if (debugState.isNodeDebugEnabled()) {
                    logger.debug(`Edge ${edge.id} has very small distance: ${distance.toFixed(6)}`);
                }
            }

            const geometry = this.createLineGeometry(source, target);
            const material = this.createEdgeMaterial();
            const line = new Mesh(geometry, material);

            // Enable both layers for the edge
            line.layers.enable(0);
            line.layers.enable(1);
            
            this.edgeGroup.add(line);
            this.edges.set(edge.id, line);
            this.edgeData.set(edge.id, edge);
            edgesCreated++;
        });
        
        // Log summary
        if (debugState.isNodeDebugEnabled()) {
            logger.debug(`Edge update complete: ${edgesCreated} edges created, ${edgesSkipped} edges skipped`);
            // Log first few edge positions for debugging
            if (this.edgeData.size > 0) this.logEdgePositions(3);
        }
    }

    /**
     * Handle settings updates for all edges
     */
    public handleSettingsUpdate(settings: Settings): void {
        this.settings = settings;
        
        // Update all edge materials
        this.edges.forEach((edge) => {
            if (edge.material instanceof LineBasicMaterial) {
                edge.material.opacity = settings.visualization.edges.opacity;
                try {
                    edge.material.color.set(settings.visualization.edges.color);
                } catch (error) {
                    console.warn('Could not update edge material color');
                }
                edge.material.needsUpdate = true;
            }
        });
    }

    /**
     * Log positions of edges for debugging
     */
    private logEdgePositions(count: number): void {
        if (!debugState.isNodeDebugEnabled()) return;
        
        let logged = 0;
        
        for (const [edgeId, edge] of this.edges.entries()) {
            if (logged >= count) break;
            
            const edgeData = this.edgeData.get(edgeId);
            if (!edgeData) continue;
            
            const sourcePos = this.nodeManager.getNodePosition(edgeData.source);
            const targetPos = this.nodeManager.getNodePosition(edgeData.target);
            
            logger.debug(`Edge ${edgeId} (${edgeData.source}->${edgeData.target}):`, {
                source: sourcePos ? [sourcePos.x.toFixed(2), sourcePos.y.toFixed(2), sourcePos.z.toFixed(2)] : 'unknown',
                target: targetPos ? [targetPos.x.toFixed(2), targetPos.y.toFixed(2), targetPos.z.toFixed(2)] : 'unknown',
                meshExists: edge ? 'yes' : 'no'
            });
            
            logged++;
        }
    }
    
    /**
     * Update edge positions based on node movements
     */
    public update(_deltaTime: number): void {
        this.updateFrameCount++;
        if (this.updateFrameCount % this.UPDATE_FREQUENCY !== 0) return;

        // Add debug logging to check edge count
        if (this.updateFrameCount % 60 === 0) {
            console.log(`[EdgeManager] Currently tracking ${this.edges.size} edges, ${this.edgeData.size} edge data entries`);
        }
        
        // Update edge positions based on current node positions
        this.edgeData.forEach((edgeData, edgeId) => {
            const edge = this.edges.get(edgeId);
            if (!edge) {
                console.warn(`[EdgeManager] Edge ${edgeId} not found in edges map`);
                return;
            }

            const sourcePos = this.nodeManager.getNodePosition(edgeData.source);
            const targetPos = this.nodeManager.getNodePosition(edgeData.target);

            // Log positions for debugging
            if (this.updateFrameCount % 120 === 0) {
                console.log(`[EdgeManager] Edge ${edgeId} positions - sourcePos: ${sourcePos ? 'found' : 'missing'}, targetPos: ${targetPos ? 'found' : 'missing'}`);
            }

            if (!sourcePos || !targetPos) {
                return;
            }
            
            // Validate positions
            if (!this.validateVector3(sourcePos) || !this.validateVector3(targetPos)) {
                if (this.updateFrameCount % 120 === 0)
                    if (debugState.isNodeDebugEnabled()) {
                        logger.warn(`Invalid vector in edge ${edgeId}: source=${JSON.stringify(sourcePos)}, target=${JSON.stringify(targetPos)}`);
                    }
                return; 
            }
                
            // Limit edge length
            const distance = sourcePos.distanceTo(targetPos);
            let finalTargetPos = targetPos.clone();
            
            if (distance > this.MAX_EDGE_LENGTH) {
                const direction = new Vector3().subVectors(targetPos, sourcePos).normalize();
                finalTargetPos = sourcePos.clone().add(direction.multiplyScalar(this.MAX_EDGE_LENGTH));
            }

            // Update the existing geometry's positions directly
            const posAttr = edge.geometry.getAttribute('position');
            if (posAttr) {
                // Create a new geometry instead of updating the existing one
                const newGeometry = this.createLineGeometry(sourcePos, finalTargetPos);
                edge.geometry.dispose();
                edge.geometry = newGeometry;
            }
            
            // Apply subtle pulsing animation if desired
            if (edge.material instanceof LineBasicMaterial) {
                const baseOpacity = this.settings.visualization.edges.opacity || 0.7;
                const pulse = Math.sin(Date.now() * 0.001) * 0.1 + 0.9;
                edge.material.opacity = baseOpacity * pulse;
                edge.material.needsUpdate = true;
            }
        });
    }

    // Updated version of update method that doesn't use forEach to avoid potential iterator issues
    /*
    public update(_deltaTime: number): void {
        this.updateFrameCount++;
        if (this.updateFrameCount % this.UPDATE_FREQUENCY !== 0) return;
        
        // Update edge positions based on current node positions
        const edgeDataEntries = Array.from(this.edgeData.entries());
        
        for (const [edgeId, edgeData] of edgeDataEntries) {
            const edge = this.edges.get(edgeId);
            if (!edge) continue;

            const sourcePos = this.nodeManager.getNodePosition(edgeData.source);
            const targetPos = this.nodeManager.getNodePosition(edgeData.target);

            if (!sourcePos || !targetPos) continue;

                // Validate positions
                if (!this.validateVector3(sourcePos) || !this.validateVector3(targetPos)) {
                    continue; 
                }
                
                // Limit edge length
                const distance = sourcePos.distanceTo(targetPos);
                let finalTargetPos = targetPos.clone();
                
                if (distance > this.MAX_EDGE_LENGTH) {
                    const direction = new Vector3().subVectors(targetPos, sourcePos).normalize();
                    finalTargetPos = sourcePos.clone().add(direction.multiplyScalar(this.MAX_EDGE_LENGTH));
                }

                // Update the existing geometry's positions directly
                const posAttr = edge.geometry.getAttribute('position');
                if (posAttr) {
                    // Update each vertex position directly
                    // Create a new geometry instead of updating the existing one
                    const newGeometry = this.createLineGeometry(sourcePos, finalTargetPos);
                    edge.geometry.dispose();
                    edge.geometry = newGeometry;
                }
                
                // Apply subtle pulsing animation if desired
                if (edge.material instanceof LineBasicMaterial) {
                    const baseOpacity = this.settings.visualization.edges.opacity || 0.7;
                    const pulse = Math.sin(Date.now() * 0.001) * 0.1 + 0.9;
                    edge.material.opacity = baseOpacity * pulse;
                    edge.material.needsUpdate = true;
                }
            }
        });
    }*/
    /**
     * Set XR mode for edge rendering
     */
    public setXRMode(enabled: boolean): void {
        if (enabled) {
            // In XR mode, only show on layer 1
            this.edgeGroup.layers.disable(0);
            this.edgeGroup.layers.enable(1);
            this.edgeGroup.traverse((child: Object3D) => {
                child.layers.disable(0);
                child.layers.enable(1);
            });
        } else {
            // In desktop mode, show on both layers
            this.edgeGroup.layers.enable(0);
            this.edgeGroup.layers.enable(1);
            this.edgeGroup.traverse((child: Object3D) => {
                child.layers.enable(0);
                child.layers.enable(1);
            });
        }
    }

    /**
     * Clean up resources
     */
    public dispose(): void {
        this.clearEdges();
        this.scene.remove(this.edgeGroup);
    }

    /**
     * Clear all edges and clean up resources
     */
    private clearEdges(): void {
        this.edges.forEach(edge => {
            if (edge) {
                // Remove from group first
                this.edgeGroup.remove(edge);
                
                // Dispose of geometry
                if (edge.geometry) {
                    edge.geometry.dispose();
                }
                
                // Dispose of material
                if (edge.material instanceof Material) {
                    edge.material.dispose();
                }
            }
        });
        this.edges.clear();
    }
}
