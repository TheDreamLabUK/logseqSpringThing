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
import { createLogger, createDataMetadata } from '../core/logger';
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
    private lastEdgeUpdateTime = 0;
    private edgeUpdateCount = 0;

    // Reusable objects to avoid allocation during updates
    private tempDirection = new Vector3();
    private tempSourceVec = new Vector3();
    private tempTargetVec = new Vector3();

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

    /**
     * Creates an optimized line geometry for an edge between two points
     */
    private createLineGeometry(source: Vector3, target: Vector3): BufferGeometry {
        const geometry = new BufferGeometry();
        
        // Apply maximum edge length limit to prevent explosion
        const distance = source.distanceTo(target);
        const finalTarget = target.clone();
        
        if (distance > this.MAX_EDGE_LENGTH) {
            // Create limited-length vector in same direction
            this.tempDirection.subVectors(target, source).normalize();
            // Direction check to prevent issues with zero vectors
            if (isNaN(this.tempDirection.x) || isNaN(this.tempDirection.y) || isNaN(this.tempDirection.z)) return geometry;
            finalTarget.copy(source).add(this.tempDirection.multiplyScalar(this.MAX_EDGE_LENGTH));
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
    public updateEdges(newEdges: Edge[]): void {
        // Record update time for debugging
        this.lastEdgeUpdateTime = performance.now();
        this.edgeUpdateCount++;
        
        logger.info(`Updating ${newEdges.length} edges (update #${this.edgeUpdateCount})`);
        
        // Track current edge IDs to detect removals
        const currentEdgeIds = new Set(this.edges.keys());
        const updatedEdgeIds = new Set<string>();
        
        // Only clear source-target cache, we'll rebuild it while keeping active edges
        this.sourceTargetCache.clear();
        
        // Track counts for debugging
        let edgesCreated = 0;
        let edgesUpdated = 0;
        let edgesSkipped = 0;
        let edgesRemoved = 0;
        let positionsFound = 0;
        
        // Log more detailed information about the edges
        if (debugState.isNodeDebugEnabled()) {
            const edgesWithPositions = newEdges.filter(e => e.sourcePosition && e.targetPosition);
            logger.debug(`Edge update details:`, createDataMetadata({
                totalEdges: newEdges.length,
                edgesWithPositions: edgesWithPositions.length,
                firstFewEdges: newEdges.slice(0, 3).map(e => ({
                    id: e.id,
                    source: e.source,
                    target: e.target,
                    hasSourcePos: !!e.sourcePosition,
                    hasTargetPos: !!e.targetPosition
                }))
            }));
        }

        // Process new edges
        newEdges.forEach(edge => {
            // Cache mapping between source and target IDs
            const edgeId = edge.id || `${edge.source}_${edge.target}`;
            updatedEdgeIds.add(edgeId);
            this.sourceTargetCache.set(edgeId, `${edge.source}:${edge.target}`);
            
            let sourcePosition = edge.sourcePosition;
            let targetPosition = edge.targetPosition;
            
            // Try to get positions from node manager if not provided
            if (!sourcePosition || !targetPosition) {
                const sourcePos = this.nodeManager.getNodePosition(edge.source);
                const targetPos = this.nodeManager.getNodePosition(edge.target);
                
                if (sourcePos && targetPos) {
                    // We found positions from the node manager, use them
                    sourcePosition = sourcePos;
                    targetPosition = targetPos;
                    positionsFound++;
                    
                    if (debugState.isNodeDebugEnabled() && edgesCreated < 3) {
                        logger.debug(`Found positions for edge ${edgeId} from node manager`);
                    }
                } else {
                    // Still missing positions, skip this edge
                    if (debugState.isNodeDebugEnabled() && edgesSkipped < 3) {
                        logger.debug(`Missing positions for edge ${edgeId}: sourcePos=${!!sourcePos}, targetPos=${!!targetPos}`);
                    }
                    edgesSkipped++;
                    return;
                }
            }

            // Clamp positions to reasonable values
            this.tempSourceVec.set(
                Math.min(100, Math.max(-100, sourcePosition.x)),
                Math.min(100, Math.max(-100, sourcePosition.y)),
                Math.min(100, Math.max(-100, sourcePosition.z))
            );
            
            this.tempTargetVec.set(
                Math.min(100, Math.max(-100, targetPosition.x)),
                Math.min(100, Math.max(-100, targetPosition.y)),
                Math.min(100, Math.max(-100, targetPosition.z))
            );

            // Create permanent copies for storage
            const source = this.tempSourceVec.clone();
            const target = this.tempTargetVec.clone();

            // Skip edge creation if source and target are too close
            const distance = this.tempSourceVec.distanceTo(this.tempTargetVec);
            if (distance < 0.001) {
                if (debugState.isNodeDebugEnabled() && edgesSkipped < 5) {
                    logger.debug(`Skipping edge ${edgeId} - nodes too close: distance=${distance.toFixed(6)}`);
                }
                edgesSkipped++;
                return;
            } else if (distance < 0.1) {
                if (debugState.isNodeDebugEnabled() && edgesCreated < 5) {
                    logger.debug(`Edge ${edgeId} has very small distance: ${distance.toFixed(6)}`);
                }
            }

            // Check if we already have this edge
            const existingEdge = this.edges.get(edgeId);
            if (existingEdge) {
                // Update existing edge - store new positions
                this.edgeData.set(edgeId, {
                    ...edge,
                    sourcePosition: source,
                    targetPosition: target
                });
                edgesUpdated++;
            } else {
                // Create new edge
                const geometry = this.createLineGeometry(source, target);
                const material = this.createEdgeMaterial();
                const line = new Mesh(geometry, material);

                // Enable both layers for the edge
                line.layers.enable(0);
                line.layers.enable(1);
                
                this.edgeGroup.add(line);
                this.edges.set(edgeId, line);
                this.edgeData.set(edgeId, {
                    ...edge,
                    sourcePosition: source,
                    targetPosition: target
                });
                edgesCreated++;
            }
        });

        // Remove edges that weren't in the update
        for (const edgeId of currentEdgeIds) {
            if (!updatedEdgeIds.has(edgeId)) {
                const edge = this.edges.get(edgeId);
                if (edge) {
                    // Remove from group
                    this.edgeGroup.remove(edge);
                    
                    // Dispose of geometry and material
                    if (edge.geometry) {
                        edge.geometry.dispose();
                    }
                    
                    if (edge.material instanceof Material) {
                        edge.material.dispose();
                    }
                    
                    // Remove from maps
                    this.edges.delete(edgeId);
                    this.edgeData.delete(edgeId);
                    edgesRemoved++;
                }
            }
        }
        
        // Log summary
        logger.info(`Edge update complete:`, createDataMetadata({
            edgesCreated,
            edgesUpdated,
            edgesSkipped,
            edgesRemoved,
            positionsFound,
            totalEdges: this.edges.size
        }));
        
        // Log first few edge positions for debugging
        if (this.edgeData.size > 0 && debugState.isNodeDebugEnabled()) {
            this.logEdgePositions(3);
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
                    logger.warn('Could not update edge material color');
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
            
            logger.debug(`Edge ${edgeId} (${edgeData.source}->${edgeData.target}):`, createDataMetadata({
                source: sourcePos ? [sourcePos.x.toFixed(2), sourcePos.y.toFixed(2), sourcePos.z.toFixed(2)] : 'unknown',
                target: targetPos ? [targetPos.x.toFixed(2), targetPos.y.toFixed(2), targetPos.z.toFixed(2)] : 'unknown',
                meshExists: edge ? 'yes' : 'no'
            }));
            
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
            logger.info(`Currently tracking ${this.edges.size} edges, ${this.edgeData.size} edge data entries, last update: ${Math.round((performance.now() - this.lastEdgeUpdateTime)/1000)}s ago`);
        }
        
        // Update edge positions based on current node positions
        this.edgeData.forEach((edgeData, edgeId) => {
            const edge = this.edges.get(edgeId);
            if (!edge) {
                if (this.updateFrameCount % 120 === 0) {
                    logger.warn(`Edge ${edgeId} not found in edges map`);
                }
                return;
            }

            const sourcePos = this.nodeManager.getNodePosition(edgeData.source);
            const targetPos = this.nodeManager.getNodePosition(edgeData.target);

            // Log positions for debugging
            if (this.updateFrameCount % 120 === 0 && debugState.isNodeDebugEnabled()) {
                logger.debug(`Edge ${edgeId} positions - sourcePos: ${sourcePos ? 'found' : 'missing'}, targetPos: ${targetPos ? 'found' : 'missing'}`);
            }

            if (!sourcePos || !targetPos) {
                return;
            }
            
            // Validate positions
            if (!this.validateVector3(sourcePos) || !this.validateVector3(targetPos)) {
                if (this.updateFrameCount % 120 === 0 && debugState.isNodeDebugEnabled()) {
                    logger.warn(`Invalid vector in edge ${edgeId}:`, createDataMetadata({
                        source: sourcePos ? [sourcePos.x, sourcePos.y, sourcePos.z] : 'invalid',
                        target: targetPos ? [targetPos.x, targetPos.y, targetPos.z] : 'invalid'
                    }));
                }
                return; 
            }
                
            // Limit edge length
            const distance = sourcePos.distanceTo(targetPos);
            
            // Reuse vectors to reduce allocations
            this.tempTargetVec.copy(targetPos);
            
            if (distance > this.MAX_EDGE_LENGTH) {
                this.tempDirection.subVectors(targetPos, sourcePos).normalize();
                this.tempTargetVec.copy(sourcePos).add(
                    this.tempDirection.multiplyScalar(this.MAX_EDGE_LENGTH)
                );
            }

            // Update the existing geometry's positions directly
            const posAttr = edge.geometry.getAttribute('position');
            if (posAttr) {
                // Update the existing BufferAttribute instead of recreating the geometry
                posAttr.setXYZ(0, sourcePos.x, sourcePos.y, sourcePos.z);
                posAttr.setXYZ(1, this.tempTargetVec.x, this.tempTargetVec.y, this.tempTargetVec.z);
                posAttr.needsUpdate = true;
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
