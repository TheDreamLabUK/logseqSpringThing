import { Logger } from './logger';
import { Node, Edge } from '../state/graphData';

const log = new Logger('GraphCache');

interface CachedGraph {
    timestamp: number;
    nodes: Node[];
    edges: Edge[];
}

export class GraphCache {
    private static readonly CACHE_KEY = 'graph_data_cache';
    private static readonly CACHE_TTL = 1000 * 60 * 5; // 5 minutes

    public static async saveToCache(nodes: Node[], edges: Edge[]): Promise<void> {
        try {
            const cacheData: CachedGraph = {
                timestamp: Date.now(),
                nodes,
                edges,
            };

            await localStorage.setItem(this.CACHE_KEY, JSON.stringify(cacheData));
            log.debug('Graph data saved to cache');
        } catch (error) {
            log.warn('Failed to save graph data to cache:', error);
        }
    }

    public static async loadFromCache(): Promise<{ nodes: Node[], edges: Edge[] } | null> {
        try {
            const cachedData = localStorage.getItem(this.CACHE_KEY);
            if (!cachedData) {
                return null;
            }

            const data: CachedGraph = JSON.parse(cachedData);
            
            // Check if cache is still valid
            if (Date.now() - data.timestamp > this.CACHE_TTL) {
                localStorage.removeItem(this.CACHE_KEY);
                return null;
            }

            // Convert plain objects back to Float32Arrays
            const nodes = data.nodes.map(node => ({
                ...node,
                position: new Float32Array(node.position),
                velocity: new Float32Array(node.velocity),
            }));

            log.debug('Graph data loaded from cache');
            return { nodes, edges: data.edges };
        } catch (error) {
            log.warn('Failed to load graph data from cache:', error);
            return null;
        }
    }

    public static clearCache(): void {
        try {
            localStorage.removeItem(this.CACHE_KEY);
            log.debug('Graph cache cleared');
        } catch (error) {
            log.warn('Failed to clear graph cache:', error);
        }
    }
} 