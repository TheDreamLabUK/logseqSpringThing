import { ControlPanel } from './ControlPanel';
import { createLogger } from '../core/logger';
import './ControlPanel.css';

const logger = createLogger('UI');

import { SettingsStore } from '../state/SettingsStore';
import { platformManager } from '../platform/platformManager';

// Initialize UI components
export async function initializeUI(): Promise<void> {
    try {
        logger.debug('Initializing UI components');
        
        const controlPanelElement = document.getElementById('control-panel');
        if (controlPanelElement instanceof HTMLElement) {
            logger.debug('Found control panel element, initializing ControlPanel');
            
            // Initialize settings first
            const settingsStore = SettingsStore.getInstance();
            await settingsStore.initialize();
            
            const controlPanel = ControlPanel.initialize(controlPanelElement);
            
            // Check if we should hide control panel based on platform
            if (platformManager.isQuest()) {
                controlPanel.hide();
            }
            
            // Settings should now be initialized
            const settings = settingsStore.get('');
            if (settings) {
                logger.debug('Applying initial settings to control panel');
                // Settings will be handled by ControlPanel's internal subscription
            } else {
                logger.error('Settings still not available after initialization');
            }
        } else {
            logger.error('Control panel element not found');
        }
    } catch (error) {
        logger.error('Failed to initialize UI:', error);
        throw error; // Re-throw to ensure caller knows initialization failed
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
