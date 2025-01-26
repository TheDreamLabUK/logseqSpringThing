import { ControlPanel } from './ControlPanel';
import { createLogger } from '../core/logger';
import { SettingsStore } from '../state/SettingsStore';
import './ControlPanel.css';

const logger = createLogger('UI');

// Initialize UI components
const initializeUI = async () => {
    try {
        logger.debug('Initializing UI components');
        
        // Wait for settings store to initialize first
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();
        logger.debug('Settings store initialized');
        
        const controlPanelElement = document.getElementById('control-panel');
        if (controlPanelElement) {
            logger.debug('Found control panel element, initializing ControlPanel');
            new ControlPanel(controlPanelElement);
        } else {
            logger.error('Control panel element not found');
        }
    } catch (error) {
        logger.error('Failed to initialize UI:', error);
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeUI);
} else {
    initializeUI();
}

export { ControlPanel };
