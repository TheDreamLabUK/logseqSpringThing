import { initializeLogger, checkLoggerConfig } from './core/logger';
import { SettingsStore } from './state/SettingsStore';
import { GraphVisualization } from './index';
import { GraphDataManager } from './state/graphData';

async function init() {
    try {
        // Initialize settings first
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();

        // Then initialize logger with settings
        await initializeLogger();
        checkLoggerConfig(); // Check logger configuration

        // Initialize graph data manager
        const graphDataManager = GraphDataManager.getInstance();
        await graphDataManager.loadInitialGraphData();

        // Now create visualization
        const visualization = new GraphVisualization();

    } catch (error) {
        console.error('Initialization failed:', error);
    }
}

init().catch(console.error); 