import { Vector3 } from 'three';
import { createLogger } from '../core/utils';
import { GraphData, Node, Edge } from '../core/types';

const logger = createLogger('GraphDataManager');

// Server-side data structures
interface ServerNode {
    id: string;
    label: string;
    metadata?: Record<string, unknown>;
    data?: {
        position?: number[];
        velocity?: { x: number; y: number; z: number };
        type?: string;
    };
    properties?: Record<string, unknown>;
}

interface ServerEdge {
    source: string;
    target: string;
    edge_type?: string;
    metadata?: Record<string, unknown>;
    properties?: Record<string, unknown>;
}

interface TransformedNode extends Omit<Node, 'data' | 'position'> {
    data: {
        position: Vector3;
        velocity: Vector3;
        type?: string;
    };
}

class ServerDataTransformer {
    transform(data: { nodes: ServerNode[]; edges: ServerEdge[]; metadata?: Record<string, unknown> }): GraphData {
        return {
            nodes: data.nodes.map(node => {
                const position = new Vector3(
                    node.data?.position?.[0] || Math.random() * 100 - 50,
                    node.data?.position?.[1] || Math.random() * 100 - 50,
                    node.data?.position?.[2] || Math.random() * 100 - 50
                );
                return {
                    id: node.id,
                    label: node.label,
                    position, // Required by Node interface
                    data: {
                        position, // Also keep in data for consistency
                        velocity: new Vector3(0, 0, 0),
                        type: node.data?.type
                    },
                    properties: node.properties || node.metadata || {}
                } as Node;
            }),
            edges: data.edges.map(edge => ({
                source: edge.source,
                target: edge.target,
                type: edge.edge_type || 'default',
                properties: edge.properties || edge.metadata || {}
            })),
            metadata: data.metadata || {}
        };
    }
}

export class GraphDataManager {
    private static instance: GraphDataManager;
    private nodes = new Map<string, TransformedNode>();
    private edges = new Map<string, Edge>();
    private currentPage = 0;
    private hasMorePages = true;
    private loadingNodes = false;
    private pageSize = 100;
    private updateListeners = new Set<(data: GraphData) => void>();
    private positionUpdateListeners = new Set<(positions: Float32Array) => void>();
    private metadata: Record<string, unknown> = {};
    private transformer: ServerDataTransformer;

    constructor() {
        this.transformer = new ServerDataTransformer();
    }

    static getInstance(): GraphDataManager {
        if (!GraphDataManager.instance) {
            GraphDataManager.instance = new GraphDataManager();
        }
        return GraphDataManager.instance;
    }

    async loadInitialGraphData(): Promise<void> {
        try {
            debugger; // Breakpoint 1: Before initial graph data load
            // Reset state
            this.nodes.clear();
            this.edges.clear();
            this.currentPage = 0;
            this.hasMorePages = true;
            this.loadingNodes = false;

            // First, update the graph data from the backend
            try {
                debugger; // Breakpoint 2: Before graph update request
                const updateResponse = await fetch('/api/graph/update', {
                    method: 'POST',
                });

                if (!updateResponse.ok) {
                    logger.warn('Graph update returned ' + updateResponse.status + ', continuing with initial load');
                } else {
                    const updateResult = await updateResponse.json();
                    debugger; // Breakpoint 3: After graph update response
                    logger.debug('Graph update result:', updateResult);
                }
            } catch (updateError) {
                logger.warn('Graph update failed, continuing with initial load:', updateError);
            }

            // Then load the first page
            await this.loadNextPage();
            
            // Notify listeners of initial data
            debugger; // Breakpoint 4: Before notifying listeners
            this.notifyUpdateListeners();

            logger.debug('Initial graph data loaded:', {
                nodes: this.nodes.size,
                edges: this.edges.size
            });
        } catch (error) {
            logger.error('Failed to load initial graph data:', error);
            // Don't throw here, allow app to continue with empty graph
            this.notifyUpdateListeners();
        }
    }

    private async loadNextPage(): Promise<void> {
        if (this.loadingNodes || !this.hasMorePages) return;

        try {
            this.loadingNodes = true;
            debugger; // Breakpoint 5: Before fetching next page
            const response = await fetch(`/api/graph/data/paginated?page=${this.currentPage}&pageSize=${this.pageSize}`);
            
            if (!response.ok) {
                throw new Error(`Failed to fetch graph data: ${response.status} ${response.statusText}`);
            }

            const serverData = await response.json();
            debugger; // Breakpoint 6: After receiving page data, before processing
            logger.debug('Received graph data:', {
                nodesCount: serverData.nodes?.length || 0,
                edgesCount: serverData.edges?.length || 0,
                totalPages: serverData.totalPages,
                currentPage: serverData.currentPage,
                metadata: serverData.metadata
            });
            
            if (!serverData.nodes || !Array.isArray(serverData.nodes)) {
                throw new Error('Invalid graph data: nodes array is missing or invalid');
            }
            
            // Transform and update graph with new nodes and edges
            const transformedData = this.transformer.transform(serverData);
            transformedData.nodes.forEach(node => {
                const transformedNode = {
                    ...node,
                    position: node.data?.position || node.position // Ensure position is at top level, fallback to node.position
                } as unknown as TransformedNode;
                this.nodes.set(node.id, transformedNode);
            });
            transformedData.edges.forEach(edge => {
                const edgeId = this.createEdgeId(edge.source, edge.target);
                this.edges.set(edgeId, edge);
            });

            debugger; // Breakpoint 7: After processing page data, before pagination update
            // Update pagination state
            this.currentPage = serverData.currentPage;
            this.hasMorePages = serverData.currentPage < serverData.totalPages;

            // Notify listeners of updated data
            this.notifyUpdateListeners();

            logger.debug('Loaded page ' + this.currentPage + ' of graph data:', {
                nodes: this.nodes.size,
                edges: this.edges.size
            });
        } catch (error) {
            logger.error('Failed to load graph data:', error);
            this.hasMorePages = false;  // Stop trying to load more pages on error
        } finally {
            this.loadingNodes = false;
        }
    }

    private createEdgeId(source: string, target: string): string {
        return `${source}-${target}`;
    }

    /**
     * Subscribe to graph data updates
     */
    subscribe(listener: (data: GraphData) => void): () => void {
        this.updateListeners.add(listener);
        return () => {
            this.updateListeners.delete(listener);
        };
    }

    /**
     * Subscribe to position updates only
     */
    subscribeToPositionUpdates(listener: (positions: Float32Array) => void): () => void {
        this.positionUpdateListeners.add(listener);
        return () => {
            this.positionUpdateListeners.delete(listener);
        };
    }

    /**
     * Get the current graph data
     */
    getGraphData(): GraphData {
        return {
            nodes: Array.from(this.nodes.values()).map(node => ({
                ...node,
                position: node.data.position // Ensure position is at top level
            })) as Node[],
            edges: Array.from(this.edges.values()),
            metadata: this.metadata
        };
    }

    private notifyUpdateListeners(): void {
        const data = this.getGraphData();
        debugger; // Breakpoint: Before notifying graph data listeners
        this.updateListeners.forEach(listener => {
            try {
                listener(data);
            } catch (error) {
                logger.error('Error in graph update listener:', error);
            }
        });
    }

    /**
     * Update graph metadata
     */
    updateMetadata(metadata: Record<string, unknown>): void {
        this.metadata = { ...this.metadata, ...metadata };
        this.notifyUpdateListeners();
    }

    /**
     * Update node positions in bulk using a Float32Array
     * This is used for physics simulation updates
     */
    updatePositions(positions: Float32Array): void {
        debugger; // Breakpoint: Before updating positions
        this.positionUpdateListeners.forEach(listener => {
            try {
                listener(positions);
            } catch (error) {
                logger.error('Error in position update listener:', error);
            }
        });
    }
}

// Export a singleton instance
export const graphDataManager = GraphDataManager.getInstance();
