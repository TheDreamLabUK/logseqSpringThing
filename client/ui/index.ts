import { ControlPanel } from './ControlPanel';
import { createLogger } from '../core/logger';
import { SettingsStore } from '../state/SettingsStore';
import './ControlPanel.css';

const logger = createLogger('UI');

// Initialize UI components
export async function initializeUI(): Promise<void> {
    try {
        logger.debug('Initializing UI components');
        
        // Get the singleton instance of SettingsStore
        const settingsStore = SettingsStore.getInstance();
        
        // Initialize settings store
        await settingsStore.initialize();
        logger.debug('Settings store initialized');
        
        const controlPanelElement = document.getElementById('control-panel');
        if (controlPanelElement instanceof HTMLElement) {
            logger.debug('Found control panel element, initializing ControlPanel');
            new ControlPanel(controlPanelElement, settingsStore);
        } else {
            logger.error('Control panel element not found');
        }
    } catch (error) {
        logger.error('Failed to initialize UI:', error);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeUI().catch(error => {
            logger.error('Failed to initialize UI on DOMContentLoaded:', error);
        });
    });
} else {
    initializeUI().catch(error => {
        logger.error('Failed to initialize UI:', error);
    });
}

export { ControlPanel };
