import { GraphData, Node, Edge, GraphDataTransformer } from '../core/types';
import { createLogger } from '../core/logger';

const logger = createLogger('GraphData');

class DefaultTransformer implements GraphDataTransformer {
    transform(data: GraphData): GraphData {
        return data;
    }
}

export class GraphDataManager {
    private nodes = new Map<string, Node>();
    private edges = new Map<string, Edge>();
    private transformer: GraphDataTransformer;

    constructor(transformer: GraphDataTransformer = new DefaultTransformer()) {
        this.transformer = transformer;
    }

    async loadData(data: GraphData): Promise<void> {
        try {
            const transformedData = this.transformer.transform(data);
            this.nodes.clear();
            this.edges.clear();

            transformedData.nodes.forEach(node => {
                this.nodes.set(node.id, node);
            });

            transformedData.edges.forEach(edge => {
                this.edges.set(`${edge.source}-${edge.target}`, edge);
            });

            logger.info('Graph data loaded', {
                nodeCount: this.nodes.size,
                edgeCount: this.edges.size
            });
        } catch (error) {
            logger.error('Failed to load graph data:', error);
            throw error;
        }
    }

    // ... rest of the implementation ...
}
