import { createLogger } from '../core/logger';
import { Vector3 } from 'three';

export class GraphDataManager {
    private static instance: GraphDataManager;
    private logger = createLogger('GraphDataManager');
    // ... other properties ...

    async loadInitialGraphData(): Promise<GraphData> {
        this.logger.debug('Loading initial graph data...');
        try {
            // Simulate data loading delay for testing
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Create some test data if none exists
            if (this.nodes.size === 0) {
                this.logger.debug('No data found, creating test data');
                // Create a simple test node
                const testNode = {
                    id: 'test1',
                    position: new Vector3(0, 0, 0),
                    size: 1
                };
                this.nodes.set(testNode.id, testNode as any);
            }

            const data = this.getGraphData();
            this.logger.debug('Initial graph data loaded:', data);
            return data;
        } catch (error) {
            this.logger.error('Failed to load initial graph data:', error);
            throw error;
        }
    }

    // ... other methods ...

    private notifyUpdateListeners(): void {
        const data = this.getGraphData();
        this.logger.debug('Notifying update listeners with data:', data);
        this.updateListeners.forEach(listener => listener(data));
    }
} 