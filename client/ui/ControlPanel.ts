import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { settingsManager } from '../state/settings';
import { WebSocketService } from '../websocket/websocketService';
import { XRSessionManager } from '../xr/xrSessionManager';
import { SceneManager } from '../rendering/scene';
import { SettingValue, getAllSettingPaths, formatSettingName, getSettingInputType } from '../types/settings/utils';
import './ControlPanel.css';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private static instance: ControlPanel | null = null;
    private container: HTMLElement;
    private settings: Settings;
    private unsubscribers: Array<() => void> = [];
    private webSocketService: WebSocketService;

    constructor(container: HTMLElement) {
        if (ControlPanel.instance) {
            throw new Error('ControlPanel is a singleton');
        }
        this.container = container;
        this.settings = {} as Settings;
        this.webSocketService = WebSocketService.getInstance();
        this.setupConnectionStatus();
        this.initializePanel();
        ControlPanel.instance = this;
    }

    public static getInstance(): ControlPanel | null {
        return ControlPanel.instance;
    }

    public show(): void {
        if (this.container) {
            this.container.style.display = 'block';
            logger.debug('Control panel shown');
        }
    }

    public hide(): void {
        if (this.container) {
            this.container.style.display = 'none';
            logger.debug('Control panel hidden');
        }
    }

    private setupConnectionStatus(): void {
        const statusElement = this.container.querySelector('.connection-status');
        const statusTextElement = this.container.querySelector('#connection-status');
        
        if (!statusElement || !statusTextElement) {
            logger.error('Connection status elements not found');
            return;
        }

        this.webSocketService.onConnectionStatusChange((connected: boolean) => {
            statusElement.classList.remove('connected', 'disconnected');
            statusElement.classList.add(connected ? 'connected' : 'disconnected');
            statusTextElement.textContent = connected ? 'Connected' : 'Disconnected';
            logger.debug('Connection status updated:', connected);
        });
    }

    private async initializePanel(): Promise<void> {
        try {
            logger.debug('Starting panel initialization...');
            
            const content = this.container.querySelector('.control-panel-content');
            if (!content) {
                throw new Error('Control panel content container not found');
            }
            
            content.innerHTML = '<div class="loading">Initializing control panel...</div>';
            
            // Initialize settings
            await settingsManager.initialize();
            content.innerHTML = '<div class="loading">Loading settings...</div>';
            
            // Get settings
            this.settings = settingsManager.getCurrentSettings();
            logger.debug('Settings loaded:', {
                visualization: Object.keys(this.settings.visualization || {}),
                system: Object.keys(this.settings.system || {}),
                xr: this.settings.xr
            });
            
            // Create panel elements
            content.innerHTML = '<div class="loading">Creating control panel...</div>';
            this.createPanelElements();
            logger.debug('Panel elements created');
            
            // Setup subscriptions
            await this.setupSettingsSubscriptions();
            logger.debug('Settings subscriptions set up');
            
            logger.info('Control panel initialized successfully');
        } catch (error) {
            logger.error('Failed to initialize control panel:', error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            this.container.innerHTML = `
                <div class="error">
                    Failed to load settings: ${errorMessage}<br>
                    <button onclick="window.location.reload()" class="retry-button">Retry</button>
                </div>
            `;
        }
    }

    private createPanelElements(): void {
        logger.debug('Creating panel elements...');
        
        if (!this.container) {
            logger.error('Control panel container is null');
            return;
        }

        const content = this.container.querySelector('.control-panel-content');
        if (!content) {
            logger.error('Control panel content container not found');
            return;
        }

        const tempContainer = document.createElement('div');
        const paths = getAllSettingPaths(this.settings);
        
        // Group settings by category
        const groupedSettings = this.groupSettingsByCategory(paths);
        const sortedCategories = Object.keys(groupedSettings).sort();

        for (const category of sortedCategories) {
            const section = this.createSection(category);
            const categorySettings = groupedSettings[category].sort();

            for (const path of categorySettings) {
                const value = settingsManager.get(path);
                const control = this.createSettingControl(path, value);
                if (control) {
                    section.appendChild(control);
                    logger.debug(`Added control for ${path}`);
                }
            }

            if (section.children.length > 1) {
                tempContainer.appendChild(section);
            }
        }

        content.innerHTML = '';
        content.appendChild(tempContainer);
    }

    private groupSettingsByCategory(paths: string[]): Record<string, string[]> {
        const groups: Record<string, string[]> = {};
        
        for (const path of paths) {
            const parts = path.split('.');
            const category = parts.slice(0, 2).join('.');
            if (!groups[category]) {
                groups[category] = [];
            }
            groups[category].push(path);
        }

        return groups;
    }

    private createSection(category: string): HTMLElement {
        const section = document.createElement('div');
        section.classList.add('settings-section');
        
        const header = document.createElement('h4');
        header.textContent = formatSettingName(category);
        section.appendChild(header);
        
        return section;
    }

    private createSettingControl(path: string, value: SettingValue): HTMLElement | null {
        // Special handling for XR mode
        if (path === 'xr.mode') {
            return this.createXRModeControl();
        }

        const control = document.createElement('div');
        control.classList.add('setting-control');

        const label = document.createElement('label');
        label.textContent = `${formatSettingName(path.split('.').pop() || '')}: `;
        control.appendChild(label);

        const input = this.createInputElement(path, value);
        if (!input) return null;

        control.appendChild(input);
        return control;
    }

    private createInputElement(path: string, value: SettingValue): HTMLElement | null {
        const type = getSettingInputType(value);
        const input = document.createElement(type === 'select' ? 'select' : 'input');
        input.dataset.path = path;

        if (input instanceof HTMLInputElement) {
            input.type = type;
            if (type === 'checkbox') {
                input.checked = value as boolean;
            } else {
                input.value = String(value);
            }
        } else if (input instanceof HTMLSelectElement && Array.isArray(value)) {
            value.forEach(opt => {
                const option = document.createElement('option');
                option.value = String(opt);
                option.text = String(opt);
                input.appendChild(option);
            });
        }

        input.addEventListener(type === 'checkbox' ? 'change' : 'input', async () => {
            try {
                const newValue = this.getInputValue(input);
                await settingsManager.updateSetting(path, newValue);
            } catch (error) {
                logger.error(`Failed to update setting ${path}:`, error);
                this.revertInput(input, value);
            }
        });

        return input;
    }

    private getInputValue(input: HTMLElement): SettingValue {
        if (input instanceof HTMLInputElement) {
            switch (input.type) {
                case 'checkbox': return input.checked;
                case 'number': return parseFloat(input.value);
                default: return input.value;
            }
        }
        if (input instanceof HTMLSelectElement) {
            return input.value;
        }
        throw new Error('Unsupported input element');
    }

    private revertInput(input: HTMLElement, originalValue: SettingValue): void {
        if (input instanceof HTMLInputElement) {
            if (input.type === 'checkbox') {
                input.checked = originalValue as boolean;
            } else {
                input.value = String(originalValue);
            }
        } else if (input instanceof HTMLSelectElement) {
            input.value = String(originalValue);
        }
    }

    private createXRModeControl(): HTMLElement {
        const control = document.createElement('div');
        control.classList.add('setting-control');

        const button = document.createElement('button');
        button.textContent = 'Enter Immersive Mode';
        button.classList.add('xr-button');
        button.addEventListener('click', async () => {
            try {
                const sceneManager = SceneManager.getInstance(
                    document.querySelector('canvas') as HTMLCanvasElement
                );
                const xrManager = XRSessionManager.getInstance(sceneManager);
                
                if (!xrManager.isXRPresenting()) {
                    await xrManager.initXRSession();
                    button.textContent = 'Exit Immersive Mode';
                } else {
                    await xrManager.endXRSession();
                    button.textContent = 'Enter Immersive Mode';
                }
            } catch (error) {
                logger.error('Failed to toggle XR session:', error);
            }
        });

        control.appendChild(button);
        return control;
    }

    private async setupSettingsSubscriptions(): Promise<void> {
        const paths = getAllSettingPaths(this.settings);
        
        for (const path of paths) {
            const unsubscribe = settingsManager.onSettingChange(path, (value) => {
                this.updateSettingControl(path, value);
            });
            this.unsubscribers.push(unsubscribe);
        }
    }

    private updateSettingControl(path: string, value: SettingValue): void {
        const input = this.container.querySelector(`[data-path="${path}"]`);
        if (!input) return;

        if (input instanceof HTMLInputElement) {
            if (input.type === 'checkbox') {
                input.checked = value as boolean;
            } else {
                input.value = String(value);
            }
        } else if (input instanceof HTMLSelectElement) {
            input.value = String(value);
        }
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsub => unsub());
        this.unsubscribers = [];
        this.webSocketService.onConnectionStatusChange(() => {});
        this.container.innerHTML = '';
        logger.debug('ControlPanel disposed');
    }
}
