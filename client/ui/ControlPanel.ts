import { SettingsStore } from '../state/SettingsStore';
import { getAllSettingPaths, formatSettingName, getSettingInputType, getStepValue } from '../types/settings/utils';
import { ValidationErrorDisplay } from '../components/settings/ValidationErrorDisplay';
import { createLogger } from '../core/logger';
import { platformManager } from '../platform/platformManager';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private static instance: ControlPanel | null = null;
    private container: HTMLDivElement;
    private settingsStore: SettingsStore;
    private validationDisplay: ValidationErrorDisplay;
    private unsubscribers: (() => void)[] = [];

    private constructor(parentElement: HTMLElement) {
        this.settingsStore = SettingsStore.getInstance();
        
        // Use existing control-panel element if it exists, otherwise create new
        const existingPanel = document.getElementById('control-panel');
        if (existingPanel instanceof HTMLDivElement) {
            this.container = existingPanel;
        } else {
            this.container = document.createElement('div');
            this.container.id = 'control-panel';
            parentElement.appendChild(this.container);
        }

        // Initialize validation error display
        this.validationDisplay = new ValidationErrorDisplay(this.container);

        // Check platform and settings before showing panel
        if (platformManager.isQuest()) {
            this.hide();
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
        // Styles are now loaded from ControlPanel.css
        return;
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
