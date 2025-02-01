import { SettingsStore } from '../state/SettingsStore';
import { getAllSettingPaths, formatSettingName, getSettingInputType, getStepValue } from '../types/settings/utils';
import { ValidationErrorDisplay } from '../components/settings/ValidationErrorDisplay';
import { createLogger } from '../core/logger';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private static instance: ControlPanel | null = null;
    private container: HTMLDivElement;
    private settingsStore: SettingsStore;
    private validationDisplay: ValidationErrorDisplay;
    private unsubscribers: (() => void)[] = [];

    private constructor(parentElement: HTMLElement) {
        this.settingsStore = SettingsStore.getInstance();
        this.container = document.createElement('div');
        this.container.className = 'settings-panel';
        parentElement.appendChild(this.container);

        // Initialize validation error display
        this.validationDisplay = new ValidationErrorDisplay(this.container);

        // Check platform and settings before showing panel
        const { platformManager } = require('../platform/platformManager');
        if (platformManager.isQuest()) {
            this.container.style.display = 'none';
        }

        this.initializePanel();
    }

    public static getInstance(): ControlPanel {
        if (!ControlPanel.instance) {
            // Create instance with document.body as default parent
            ControlPanel.instance = new ControlPanel(document.body);
        }
        return ControlPanel.instance;
    }

    public static initialize(parentElement: HTMLElement): ControlPanel {
        if (!ControlPanel.instance) {
            ControlPanel.instance = new ControlPanel(parentElement);
        }
        return ControlPanel.instance;
    }

    private async initializePanel(): Promise<void> {
        try {
            await this.settingsStore.initialize();
            
            // Get all setting paths
            const settings = this.settingsStore.get('') as any;
            const paths = getAllSettingPaths(settings);
            
            // Group settings by category
            const groupedSettings = this.groupSettingsByCategory(paths);
            
            // Create sections for each category
            for (const [category, paths] of Object.entries(groupedSettings)) {
                const section = await this.createSection(category, paths);
                this.container.appendChild(section);
            }

            // Add styles
            this.addStyles();
            
            logger.info('Control panel initialized');
        } catch (error) {
            logger.error('Failed to initialize control panel:', error);
        }
    }

    private groupSettingsByCategory(paths: string[]): Record<string, string[]> {
        const groups: Record<string, string[]> = {};
        
        paths.forEach(path => {
            const [category] = path.split('.');
            if (!groups[category]) {
                groups[category] = [];
            }
            groups[category].push(path);
        });
        
        return groups;
    }

    private async createSection(category: string, paths: string[]): Promise<HTMLElement> {
        const section = document.createElement('div');
        section.className = 'settings-section';
        
        const header = document.createElement('h2');
        header.textContent = formatSettingName(category);
        header.className = 'settings-section-header';
        section.appendChild(header);
        
        // Group paths by subcategory
        const subcategories = this.groupBySubcategory(paths);
        
        for (const [subcategory, subPaths] of Object.entries(subcategories)) {
            const subsection = await this.createSubsection(subcategory, subPaths);
            section.appendChild(subsection);
        }
        
        return section;
    }

    private groupBySubcategory(paths: string[]): Record<string, string[]> {
        const groups: Record<string, string[]> = {};
        
        paths.forEach(path => {
            const parts = path.split('.');
            if (parts.length > 2) {
                const subcategory = parts[1];
                if (!groups[subcategory]) {
                    groups[subcategory] = [];
                }
                groups[subcategory].push(path);
            } else {
                if (!groups['general']) {
                    groups['general'] = [];
                }
                groups['general'].push(path);
            }
        });
        
        return groups;
    }

    private async createSubsection(subcategory: string, paths: string[]): Promise<HTMLElement> {
        const subsection = document.createElement('div');
        subsection.className = 'settings-subsection';
        
        const header = document.createElement('h3');
        header.textContent = formatSettingName(subcategory);
        header.className = 'settings-subsection-header';
        subsection.appendChild(header);
        
        for (const path of paths) {
            const control = await this.createSettingControl(path);
            subsection.appendChild(control);
        }
        
        return subsection;
    }

    private async createSettingControl(path: string): Promise<HTMLElement> {
        const container = document.createElement('div');
        container.className = 'setting-control';
        
        const label = document.createElement('label');
        label.textContent = formatSettingName(path.split('.').pop() || '');
        container.appendChild(label);
        
        const value = this.settingsStore.get(path);
        const input = this.createInputElement(path, value);
        container.appendChild(input);
        
        // Subscribe to changes
        const unsubscribe = await this.settingsStore.subscribe(path, (_, newValue) => {
            this.updateInputValue(input, newValue);
        });
        this.unsubscribers.push(unsubscribe);
        
        return container;
    }

    private createInputElement(path: string, value: any): HTMLElement {
        const inputType = getSettingInputType(value);
        let input: HTMLElement;
        
        switch (inputType) {
            case 'checkbox':
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'checkbox';
                (input as HTMLInputElement).checked = value as boolean;
                input.addEventListener('change', async (e) => {
                    const target = e.target as HTMLInputElement;
                    try {
                        await this.settingsStore.set(path, target.checked);
                    } catch (error) {
                        logger.error(`Failed to update ${path}:`, error);
                        // Revert the checkbox state
                        target.checked = !target.checked;
                    }
                });
                break;
                
            case 'number':
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'number';
                (input as HTMLInputElement).value = String(value);
                (input as HTMLInputElement).step = getStepValue(path);
                input.addEventListener('change', async (e) => {
                    const target = e.target as HTMLInputElement;
                    try {
                        await this.settingsStore.set(path, Number(target.value));
                    } catch (error) {
                        logger.error(`Failed to update ${path}:`, error);
                        // Revert the input value
                        target.value = String(value);
                    }
                });
                break;
                
            case 'color':
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'color';
                (input as HTMLInputElement).value = value as string;
                input.addEventListener('change', async (e) => {
                    const target = e.target as HTMLInputElement;
                    try {
                        await this.settingsStore.set(path, target.value);
                    } catch (error) {
                        logger.error(`Failed to update ${path}:`, error);
                        // Revert the color
                        target.value = value as string;
                    }
                });
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
                input.addEventListener('change', async (e) => {
                    const target = e.target as HTMLSelectElement;
                    try {
                        await this.settingsStore.set(path, target.value);
                    } catch (error) {
                        logger.error(`Failed to update ${path}:`, error);
                        // Revert the selection
                        target.value = value as string;
                    }
                });
                break;
                
            default:
                input = document.createElement('input');
                (input as HTMLInputElement).type = 'text';
                (input as HTMLInputElement).value = String(value);
                input.addEventListener('change', async (e) => {
                    const target = e.target as HTMLInputElement;
                    try {
                        await this.settingsStore.set(path, target.value);
                    } catch (error) {
                        logger.error(`Failed to update ${path}:`, error);
                        // Revert the input value
                        target.value = String(value);
                    }
                });
        }
        
        input.className = 'setting-input';
        return input;
    }

    private updateInputValue(input: HTMLElement, value: any): void {
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

    private addStyles(): void {
        const style = document.createElement('style');
        style.textContent = `
            .settings-panel {
                padding: 20px;
                background: #f5f5f5;
                border-radius: 8px;
                max-width: 800px;
                margin: 0 auto;
            }

            .settings-section {
                margin-bottom: 24px;
                background: white;
                padding: 16px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .settings-section-header {
                margin: 0 0 16px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid #eee;
                color: #333;
                font-size: 18px;
            }

            .settings-subsection {
                margin: 16px 0;
                padding: 16px;
                background: #f9f9f9;
                border-radius: 4px;
            }

            .settings-subsection-header {
                margin: 0 0 12px 0;
                color: #666;
                font-size: 16px;
            }

            .setting-control {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 8px 0;
                padding: 8px;
                border-radius: 4px;
            }

            .setting-control:hover {
                background: #f0f0f0;
            }

            .setting-control label {
                flex: 1;
                margin-right: 16px;
                color: #444;
            }

            .setting-input {
                padding: 4px 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }

            .setting-input[type="checkbox"] {
                width: 20px;
                height: 20px;
            }

            .setting-input[type="color"] {
                padding: 0;
                width: 40px;
                height: 24px;
            }

            .setting-input[type="number"] {
                width: 80px;
            }

            .setting-input:focus {
                outline: none;
                border-color: #007bff;
                box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
            }
        `;
        document.head.appendChild(style);
    }

    public show(): void {
        this.container.classList.remove('hidden');
    }

    public hide(): void {
        this.container.classList.add('hidden');
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.validationDisplay.dispose();
        this.container.remove();
        ControlPanel.instance = null;
    }
}
