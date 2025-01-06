import { initializeLogger, checkLoggerConfig } from './core/logger';
import { SettingsStore } from './state/SettingsStore';
import { GraphVisualization } from './index';
import { GraphDataManager } from './state/graphData';

async function init() {
    try {
        console.log('Starting initialization...');

        // Initialize settings first
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();
        console.log('Settings initialized');

        // Then initialize logger with settings
        await initializeLogger();
        checkLoggerConfig();
        console.log('Logger initialized');

        // Create visualization with settings
        const visualization = new GraphVisualization(settingsStore.get());
        console.log('Visualization created');

    } catch (error) {
        console.error('Initialization failed:', error);
    }
}

init().catch(console.error); 