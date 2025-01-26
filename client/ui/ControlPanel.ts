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

    constructor(container: HTMLElement, settingsStore: SettingsStore) {
        this.container = container;
        this.settingsStore = settingsStore;
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
            content.parentNode?.insertBefore(header, content);
        }

        logger.debug('Panel elements created');
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
        this.container.innerHTML = '';
    }
}
