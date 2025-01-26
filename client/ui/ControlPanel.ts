import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { SettingsStore } from '../state/SettingsStore';
import { WebSocketService } from '../websocket/websocketService';
import './ControlPanel.css';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private settings: Settings;
    private unsubscribers: Array<() => void> = [];
    private settingsStore: SettingsStore;
    private webSocketService: WebSocketService;

    constructor(container: HTMLElement, settingsStore: SettingsStore) {
        this.container = container;
        this.settingsStore = settingsStore;
        this.settings = {} as Settings;
        this.webSocketService = WebSocketService.getInstance();
        this.setupConnectionStatus();
        this.initializePanel();
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
            
            // Get the content container
            const content = this.container.querySelector('.control-panel-content');
            if (!content) {
                throw new Error('Control panel content container not found');
            }
            
            // Show loading state
            content.innerHTML = '<div class="loading">Initializing control panel...</div>';
            logger.debug('Container exists:', !!this.container);
            
            // Initialize settings store with timeout
            const initializePromise = this.settingsStore.initialize();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Settings initialization timed out')), 10000);
            });
            
            // Wait for settings store to initialize or timeout
            await Promise.race([initializePromise, timeoutPromise]);
            content.innerHTML = '<div class="loading">Loading settings...</div>';
            
            // Get settings after initialization
            const settings = this.settingsStore.get('');
            if (!settings) {
                throw new Error('Failed to get settings after initialization');
            }
            
            this.settings = settings as Settings;
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
        
        // First check if the container exists
        if (!this.container) {
            logger.error('Control panel container is null');
            return;
        }

        // Get the content container
        const content = this.container.querySelector('.control-panel-content');
        if (!content) {
            logger.error('Control panel content container not found');
            return;
        }

        // Create a temporary container for the new content
        const tempContainer = document.createElement('div');

        // Create settings sections
        const flatSettings = this.flattenSettings(this.settings);
        logger.debug('Flattened settings:', {
            count: Object.keys(flatSettings).length,
            paths: Object.keys(flatSettings)
        });
        
        const groupedSettings = this.groupSettingsByCategory(flatSettings);
        logger.debug('Grouped settings:', {
            categories: Object.keys(groupedSettings),
            settingsPerCategory: Object.entries(groupedSettings).map(([cat, settings]) => ({
                category: cat,
                count: Object.keys(settings).length
            }))
        });

        // Sort categories to ensure consistent order
        const sortedCategories = Object.entries(groupedSettings).sort(([a], [b]) => a.localeCompare(b));
        logger.debug('Processing categories:', sortedCategories.map(([cat]) => ({
            category: cat,
            settingCount: Object.keys(groupedSettings[cat]).length
        })));

        for (const [category, settings] of sortedCategories) {
            const section = this.createSection(category);
            
            // Sort settings within each category
            const sortedSettings = Object.entries(settings).sort(([a], [b]) => a.localeCompare(b));
            
            for (const [path, value] of sortedSettings) {
                logger.debug(`Creating control for ${path}:`, {
                    type: typeof value,
                    value: value
                });
                const control = this.createSettingControl(path, value);
                if (control) {
                    section.appendChild(control);
                    logger.debug(`Added control for ${path}`);
                } else {
                    logger.warn(`Failed to create control for ${path}`);
                }
            }
            
            if (section.children.length > 1) { // > 1 because section always has a header
                tempContainer.appendChild(section);
                logger.debug(`Added section ${category}`, {
                    totalControls: section.children.length - 1,
                    paths: Array.from(section.querySelectorAll('[data-path]')).map(el => el.getAttribute('data-path'))
                });
            } else {
                logger.debug(`Skipping empty section ${category}`);
            }
        }

        // Only update the content once everything is ready
        content.innerHTML = '';
        content.appendChild(tempContainer);
        logger.debug('Panel elements created successfully', {
            totalSections: tempContainer.children.length,
            totalControls: tempContainer.querySelectorAll('[data-path]').length
        });

        logger.debug('Panel elements creation complete');
    }

    private flattenSettings(settings: Settings): Record<string, unknown> {
        const result: Record<string, unknown> = {};
        
        function flatten(obj: unknown, prefix = ''): void {
            if (typeof obj === 'object' && obj !== null) {
                for (const [key, value] of Object.entries(obj)) {
                    const newPrefix = prefix ? `${prefix}.${key}` : key;
                    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                        flatten(value, newPrefix);
                    } else {
                        result[newPrefix] = value;
                    }
                }
            }
        }
        
        flatten(settings);
        return result;
    }

    private groupSettingsByCategory(flatSettings: Record<string, unknown>): Record<string, Record<string, unknown>> {
        const result: Record<string, Record<string, unknown>> = {};
        
        // Group settings by their category path
        for (const [path, value] of Object.entries(flatSettings)) {
            const parts = path.split('.');
            if (parts.length >= 2) {
                // For visualization and system settings, use first two parts as category
                // e.g., 'visualization.nodes' or 'system.network'
                const category = parts.slice(0, 2).join('.');
                if (!result[category]) {
                    result[category] = {};
                }
                result[category][path] = value;
            } else {
                // For top-level settings like 'xr', use the first part
                const category = parts[0];
                if (!result[category]) {
                    result[category] = {};
                }
                result[category][path] = value;
            }
        }

        logger.debug('Grouped settings by category:', {
            categories: Object.keys(result),
            settingsPerCategory: Object.entries(result).map(([cat, settings]) => ({
                category: cat,
                settingCount: Object.keys(settings).length,
                example: Object.keys(settings)[0]
            }))
        });
        
        return result;
    }

    private createSection(category: string): HTMLElement {
        const section = document.createElement('div');
        section.classList.add('settings-section');
        
        const header = document.createElement('h4');
        header.textContent = category;
        section.appendChild(header);
        
        return section;
    }

    private createSettingControl(path: string, value: unknown): HTMLElement | null {
        const label = document.createElement('label');
        // Use the last part of the path and format it
        const labelText = path.split('.').pop() || path;
        label.textContent = `${this.formatSettingLabel(labelText)}:`;  // Use the formatting
        label.style.whiteSpace = 'nowrap';

        const control = document.createElement('div');
        control.classList.add('setting-control');
        control.appendChild(label);

        // Create input element based on value type
        if (typeof value === 'boolean') {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = value;
            checkbox.dataset.path = path;
            checkbox.addEventListener('change', () => {
                this.settingsStore.set(path, checkbox.checked);
                logger.debug(`Setting ${path} updated to:`, checkbox.checked);
            });
            control.appendChild(checkbox);
        } else if (typeof value === 'number') {
            const numberInput = document.createElement('input');
            numberInput.type = 'number';
            numberInput.value = value.toString();
            numberInput.dataset.path = path;
            numberInput.addEventListener('input', () => {
                const parsedValue = parseFloat(numberInput.value);
                if (!isNaN(parsedValue)) {
                    this.settingsStore.set(path, parsedValue);
                    logger.debug(`Setting ${path} updated to:`, parsedValue);
                }
            });
            control.appendChild(numberInput);
        } else if (typeof value === 'string' && (path.endsWith('Color') || path.includes('.color'))) {
            const colorInput = document.createElement('input');
            colorInput.type = 'color';
            colorInput.value = value;
            colorInput.dataset.path = path;
            colorInput.addEventListener('change', () => {
                this.settingsStore.set(path, colorInput.value);
                logger.debug(`Setting ${path} updated to:`, colorInput.value);
            });
            control.appendChild(colorInput);
        } else if (typeof value === 'string') {
            const textInput = document.createElement('input');
            textInput.type = 'text';
            textInput.value = value;
            textInput.dataset.path = path;
            textInput.addEventListener('input', () => {
                this.settingsStore.set(path, textInput.value);
                logger.debug(`Setting ${path} updated to:`, textInput.value);
            });
            control.appendChild(textInput);
        } else if (Array.isArray(value)) {
            const select = document.createElement('select');
            select.dataset.path = path;
            value.forEach(optionValue => {
                const option = document.createElement('option');
                option.value = String(optionValue);
                option.text = String(optionValue);
                select.appendChild(option);
            });
            select.addEventListener('change', () => {
                this.settingsStore.set(path, select.value);
                logger.debug(`Setting ${path} updated to:`, select.value);
            });
            control.appendChild(select);
        } else {
            logger.warn(`Unsupported setting type for ${path}:`, typeof value);
            return null;
        }

        return control;
    }

    private async setupSettingsSubscriptions(): Promise<void> {
        const flatSettings = this.flattenSettings(this.settings);
        const paths = Object.keys(flatSettings).sort();

        const promises = paths.map(async path => {
            return new Promise<void>((resolve, reject) => {
                const timeoutId = setTimeout(() => {
                    reject(new Error(`Timeout setting up subscription for ${path}`));
                }, 5000);

                // Subscribe to each setting path
                this.settingsStore.subscribe(path, (value: unknown) => {
                    logger.debug(`Received update for ${path}:`, value);
                    this.updateSettingControl(path, value);
                    clearTimeout(timeoutId);
                    resolve();
                }).then(unsub => {
                    this.unsubscribers.push(unsub);
                }).catch(error => {
                    logger.error(`Failed to subscribe to ${path}:`, error);
                    reject(error);
                });
            });
        });

        try {
            await Promise.all(promises);
            logger.info('All settings subscriptions set up successfully.');
        } catch (error) {
            logger.error('Failed to set up all settings subscriptions:', error);
        }
    }

    private updateSettingControl(path: string, value: unknown): void {
        try {
            const control = this.container.querySelector(`[data-path="${path}"]`);
            if (!control) {
                logger.warn(`Control not found for setting: ${path}`);
                return;
            }

            if (control instanceof HTMLInputElement) {
                const inputType = control.type;
                switch (inputType) {
                    case 'checkbox':
                        control.checked = value as boolean;
                        logger.debug(`Updated checkbox: ${path}`, { checked: control.checked });
                        break;
                    case 'number':
                        control.value = String(value);
                        logger.debug(`Updated number: ${path}`, { value: control.value });
                        break;
                    case 'color':
                        control.value = value as string;
                        logger.debug(`Updated color: ${path}`, { value: control.value });
                        break;
                    default:
                        control.value = String(value);
                        logger.debug(`Updated text: ${path}`, { value: control.value });
                }
            } else if (control instanceof HTMLSelectElement) {
                control.value = String(value);
                logger.debug(`Updated select: ${path}`, { value: control.value });
            }
        } catch (error) {
            logger.error(`Failed to update setting value: ${path}`, error);
        }
    }

    private formatSettingLabel(key: string): string {
        return key
            .split(/(?=[A-Z])|[_\.]/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsub => unsub());
        this.unsubscribers = [];
        this.webSocketService.onConnectionStatusChange(() => {}); // Remove connection status handler
        this.container.innerHTML = '';
        logger.debug('ControlPanel disposed');
    }
}
