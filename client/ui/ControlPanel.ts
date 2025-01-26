import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { SettingsStore } from '../state/SettingsStore';
import './ControlPanel.css';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private settings: Settings;
    private unsubscribers: Array<() => void> = [];
    private settingsStore: SettingsStore;

    constructor(container: HTMLElement) {
        this.container = container;
        this.settingsStore = SettingsStore.getInstance();
        this.settings = {} as Settings;
        this.initializePanel();
    }

    private async initializePanel(): Promise<void> {
        try {
            // Show loading state with more detail
            this.container.innerHTML = '<div class="loading">Initializing control panel...</div>';
            
            // Initialize settings store with timeout
            const initializePromise = this.settingsStore.initialize();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Settings initialization timed out')), 10000);
            });
            
            // Wait for settings store to initialize or timeout
            await Promise.race([initializePromise, timeoutPromise]);
            this.container.innerHTML = '<div class="loading">Loading settings...</div>';
            
            // Get settings after initialization
            const settings = this.settingsStore.get('');
            if (!settings) {
                throw new Error('Failed to get settings after initialization');
            }
            
            this.settings = settings as Settings;
            logger.info('Settings loaded:', this.settings);
            
            // Create panel elements
            this.container.innerHTML = '<div class="loading">Creating control panel...</div>';
            this.createPanelElements();
            
            // Setup subscriptions
            await this.setupSettingsSubscriptions();
            
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
        // Clear existing content but keep the header
        const header = this.container.querySelector('.control-panel-header');
        const content = this.container.querySelector('.control-panel-content');
        if (!content) {
            logger.error('Control panel content container not found');
            return;
        }
        content.innerHTML = '';

        // Create settings sections
        const flatSettings = this.flattenSettings(this.settings);
        const groupedSettings = this.groupSettingsByCategory(flatSettings);

        // Sort categories to ensure consistent order
        const sortedCategories = Object.entries(groupedSettings).sort(([a], [b]) => a.localeCompare(b));

        for (const [category, settings] of sortedCategories) {
            const section = this.createSection(category);
            
            // Sort settings within each category
            const sortedSettings = Object.entries(settings).sort(([a], [b]) => a.localeCompare(b));
            
            for (const [path, value] of sortedSettings) {
                const control = this.createSettingControl(path, value);
                if (control) {
                    section.appendChild(control);
                }
            }
            
            if (section.children.length > 1) { // > 1 because section always has a header
                content.appendChild(section);
            }
        }

        // Update connection status if header exists
        if (header) {
            const statusElement = header.querySelector('#connection-status');
            if (statusElement) {
                const connected = this.settingsStore.isInitialized();
                statusElement.textContent = connected ? 'Connected' : 'Disconnected';
                statusElement.className = connected ? 'connection-status connected' : 'connection-status disconnected';
            }
        }

        logger.debug('Panel elements created');
    }

    private flattenSettings(obj: unknown, prefix: string = ''): Record<string, unknown> {
        const result: Record<string, unknown> = {};
        
        if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
            for (const [key, value] of Object.entries(obj)) {
                const newKey = prefix ? `${prefix}.${key}` : key;
                
                if (value && typeof value === 'object' && !Array.isArray(value)) {
                    Object.assign(result, this.flattenSettings(value, newKey));
                } else {
                    result[newKey] = value;
                }
            }
        }
        
        return result;
    }

    private groupSettingsByCategory(flatSettings: Record<string, unknown>): Record<string, Record<string, unknown>> {
        const result: Record<string, Record<string, unknown>> = {};
        
        // Initialize main categories
        const mainCategories = ['visualization', 'system'];
        mainCategories.forEach(category => {
            result[category] = {};
        });
        
        // Group settings by their full category path (e.g., 'visualization.nodes')
        for (const [path, value] of Object.entries(flatSettings)) {
            const parts = path.split('.');
            if (parts.length >= 2) {
                const mainCategory = parts[0];
                const subCategory = parts[1];
                const fullCategory = `${mainCategory}.${subCategory}`;
                
                if (!result[fullCategory]) {
                    result[fullCategory] = {};
                }
                result[fullCategory][path] = value;
            } else {
                // Handle top-level settings if any
                const category = parts[0];
                if (!result[category]) {
                    result[category] = {};
                }
                result[category][path] = value;
            }
        }
        
        // Remove empty categories
        Object.keys(result).forEach(category => {
            if (Object.keys(result[category]).length === 0) {
                delete result[category];
            }
        });
        
        return result;
    }

    private createSection(category: string): HTMLElement {
        const section = document.createElement('div');
        section.className = 'settings-section';
        
        const header = document.createElement('h2');
        
        // Handle nested categories (e.g., 'visualization.nodes')
        const parts = category.split('.');
        if (parts.length === 2) {
            // Format as "Nodes Settings" for visualization.nodes
            const mainCategory = this.formatCategoryName(parts[0]);
            const subCategory = this.formatCategoryName(parts[1]);
            header.textContent = `${subCategory} Settings`;
            
            // Add a subtitle with the main category
            const subtitle = document.createElement('span');
            subtitle.className = 'settings-subtitle';
            subtitle.textContent = mainCategory;
            header.appendChild(subtitle);
        } else {
            // For top-level categories, just format the name
            header.textContent = this.formatCategoryName(category);
        }
        
        section.appendChild(header);
        return section;
    }

    private createSettingControl(path: string, value: unknown): HTMLElement | null {
        const container = document.createElement('div');
        container.className = 'setting-control';
        
        const label = document.createElement('label');
        label.textContent = this.formatSettingName(path.split('.').pop()!);
        container.appendChild(label);
        
        const control = this.createInputElement(path, value);
        if (!control) {
            return null;
        }
        
        container.appendChild(control);
        return container;
    }

    private createInputElement(path: string, value: unknown): HTMLElement | null {
        const type = this.getInputType(value);
        if (!type) {
            return null;
        }

        let input: HTMLElement;
        
        switch (type) {
            case 'checkbox':
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'checkbox';
                (input as HTMLInputElement).checked = value as boolean;
                input.onchange = (e: Event) => {
                    const target = e.target as HTMLInputElement;
                    this.settingsStore.set(path, target.checked);
                };
                break;

            case 'number':
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'number';
                (input as HTMLInputElement).value = String(value);
                (input as HTMLInputElement).step = this.getStepValue(path);
                input.onchange = (e: Event) => {
                    const target = e.target as HTMLInputElement;
                    this.settingsStore.set(path, parseFloat(target.value));
                };
                break;

            case 'color':
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'color';
                (input as HTMLInputElement).value = value as string;
                input.onchange = (e: Event) => {
                    const target = e.target as HTMLInputElement;
                    this.settingsStore.set(path, target.value);
                };
                break;

            case 'select':
                input = document.createElement('select');
                if (Array.isArray(value)) {
                    value.forEach(option => {
                        const opt = document.createElement('option');
                        opt.value = String(option);
                        opt.textContent = String(option);
                        input.appendChild(opt);
                    });
                }
                input.onchange = (e: Event) => {
                    const target = e.target as HTMLSelectElement;
                    this.settingsStore.set(path, target.value);
                };
                break;

            default:
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'text';
                (input as HTMLInputElement).value = String(value);
                input.onchange = (e: Event) => {
                    const target = e.target as HTMLInputElement;
                    this.settingsStore.set(path, target.value);
                };
        }

        input.id = `setting-${path}`;
        return input;
    }

    private getInputType(value: unknown): string | null {
        switch (typeof value) {
            case 'boolean':
                return 'checkbox';
            case 'number':
                return 'number';
            case 'string':
                if (value.match(/^#[0-9a-f]{6}$/i)) {
                    return 'color';
                }
                return 'text';
            case 'object':
                if (Array.isArray(value)) {
                    return 'select';
                }
                return null;
            default:
                return null;
        }
    }

    private getStepValue(path: string): string {
        if (path.includes('opacity') || path.includes('strength')) {
            return '0.1';
        }
        return '1';
    }

    private formatCategoryName(category: string): string {
        return category
            .split(/(?=[A-Z])/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    private formatSettingName(setting: string): string {
        // Get the last part of the path (e.g., 'baseSize' from 'visualization.nodes.baseSize')
        const parts = setting.split('.');
        const name = parts[parts.length - 1];
        
        // Convert camelCase to Title Case with spaces
        return name
            .split(/(?=[A-Z])/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    private async setupSettingsSubscriptions(): Promise<void> {
        try {
            logger.debug('Setting up settings subscriptions');
            
            // Clear existing subscriptions
            this.unsubscribers.forEach(unsub => unsub());
            this.unsubscribers = [];

            const settings = this.settingsStore;

            // Subscribe to all settings changes
            const flatSettings = this.flattenSettings(this.settings);
            for (const path of Object.keys(flatSettings)) {
                try {
                    const unsub = await settings.subscribe(path, (value) => {
                        logger.debug(`Setting updated: ${path}`, value);
                        this.updateSettingValue(path, value);
                        
                        // Special handling for label visibility
                        if (path === 'visualization.labels.enableLabels') {
                            this.updateLabelVisibility(typeof value === 'boolean' ? value : value === 'true');
                        }
                        
                        // Special handling for connection status
                        if (path === 'system.network.status') {
                            // Convert value to boolean, handling different possible types
                            const isConnected = typeof value === 'boolean' ? value :
                                              typeof value === 'string' ? value.toLowerCase() === 'true' :
                                              Boolean(value);
                            this.updateConnectionStatus(isConnected);
                        }
                    });
                    
                    if (unsub) {
                        this.unsubscribers.push(unsub);
                    }
                } catch (error) {
                    logger.error(`Failed to subscribe to setting: ${path}`, error);
                }
            }

            logger.debug('Settings subscriptions setup complete');
        } catch (error) {
            logger.error('Failed to setup settings subscriptions:', error);
            throw error;
        }
    }

    private updateConnectionStatus(connected: boolean): void {
        const statusElement = this.container.querySelector('#connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
        }
    }

    private updateLabelVisibility(value: boolean): void {
        // Update label visibility in the UI
        const labelElements = document.querySelectorAll('.node-label');
        labelElements.forEach(el => {
            (el as HTMLElement).style.display = value ? 'block' : 'none';
        });
    }

    private updateSettingValue(path: string, value: unknown): void {
        logger.debug(`Updating setting value: ${path}`, { value });
        
        const element = document.getElementById(`setting-${path}`);
        if (!element) {
            logger.warn(`No element found for setting: ${path}`);
            return;
        }

        try {
            if (element instanceof HTMLInputElement) {
                switch (element.type) {
                    case 'checkbox':
                        element.checked = value as boolean;
                        logger.debug(`Updated checkbox: ${path}`, { checked: element.checked });
                        break;
                    case 'number':
                        element.value = String(value);
                        logger.debug(`Updated number: ${path}`, { value: element.value });
                        break;
                    case 'color':
                        element.value = value as string;
                        logger.debug(`Updated color: ${path}`, { value: element.value });
                        break;
                    default:
                        element.value = String(value);
                        logger.debug(`Updated text: ${path}`, { value: element.value });
                }
            } else if (element instanceof HTMLSelectElement) {
                element.value = String(value);
                logger.debug(`Updated select: ${path}`, { value: element.value });
            }
        } catch (error) {
            logger.error(`Failed to update setting value: ${path}`, error);
        }
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsub => unsub());
        this.unsubscribers = [];
        this.container.innerHTML = '';
    }
}
