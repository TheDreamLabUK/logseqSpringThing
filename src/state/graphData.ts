import { createLogger } from '../core/logger';

export class GraphDataManager {
    private static instance: GraphDataManager;
    private logger = createLogger('GraphDataManager');
    // ... other properties ...

    async loadInitialGraphData(): Promise<void> {
        try {
            this.logger.debug('Loading initial graph data...');
            
            // Simulate data loading delay for testing
            await new Promise(resolve => setTimeout(resolve, 2000));

            this.logger.info('Initial graph data loaded.');
            this.hasMorePages = false;
            this.loadingNodes = false;
            this.notifyUpdateListeners();
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