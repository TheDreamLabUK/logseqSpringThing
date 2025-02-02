import { SettingsStore } from '../state/SettingsStore';
import { getAllSettingPaths, formatSettingName, getStepValue } from '../types/settings/utils';
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
        
        // Create main container
        this.container = document.createElement('div');
        this.container.id = 'control-panel';
        parentElement.appendChild(this.container);

        // Create toggle tab
        const toggleTab = document.createElement('div');
        toggleTab.className = 'panel-toggle';
        toggleTab.addEventListener('click', () => this.togglePanel());
        parentElement.appendChild(toggleTab);

        // Initialize validation error display
        this.validationDisplay = new ValidationErrorDisplay(this.container);

        // Check platform and settings before showing panel
        if (platformManager.isQuest()) {
            this.hide();
        }

        this.initializePanel();
    }

    private togglePanel() {
        this.container.classList.toggle('visible');
    }

    public static getInstance(): ControlPanel {
        if (!ControlPanel.instance) {
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
        
        const header = document.createElement('div');
        header.className = 'section-header';
        header.addEventListener('click', () => {
            header.classList.toggle('expanded');
            const content = section.querySelector('.section-content');
            if (content) {
                content.classList.toggle('expanded');
            }
        });
        
        const title = document.createElement('h4');
        title.textContent = formatSettingName(category);
        header.appendChild(title);
        section.appendChild(header);
        
        // Group paths by subcategory
        const subcategories = this.groupBySubcategory(paths);
        
        const content = document.createElement('div');
        content.className = 'section-content';
        
        for (const [subcategory, subPaths] of Object.entries(subcategories)) {
            const subsection = await this.createSubsection(subcategory, subPaths);
            content.appendChild(subsection);
        }
        
        section.appendChild(content);
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
        
        const labelContainer = document.createElement('div');
        labelContainer.className = 'setting-label';
        
        const label = document.createElement('span');
        label.textContent = formatSettingName(path.split('.').pop() || '');
        labelContainer.appendChild(label);
        
        const value = this.settingsStore.get(path);
        const valueDisplay = document.createElement('span');
        valueDisplay.className = 'setting-value';
        labelContainer.appendChild(valueDisplay);
        
        container.appendChild(labelContainer);
        
        const input = this.createInputElement(path, value, valueDisplay);
        container.appendChild(input);
        
        // Subscribe to changes
        const unsubscribe = await this.settingsStore.subscribe(path, (_, newValue) => {
            this.updateInputValue(input, newValue, valueDisplay);
        });
        this.unsubscribers.push(unsubscribe);
        
        return container;
    }

    private createInputElement(path: string, value: any, valueDisplay: HTMLElement): HTMLElement {
        const inputType = this.getInputTypeForSetting(path, value);
        let input: HTMLElement;
        
        switch (inputType) {
            case 'toggle':
                input = this.createToggleSwitch(path, value as boolean);
                break;
            case 'slider':
                input = this.createSlider(path, value as number, valueDisplay);
                break;
            case 'select':
                input = this.createSelect(path, value);
                break;
            case 'color':
                input = this.createColorPicker(path, value as string);
                break;
            default:
                input = this.createTextInput(path, value);
        }
        
        return input;
    }

    private getInputTypeForSetting(path: string, value: any): string {
        // XR mode and space type should be dropdowns
        if (path.endsWith('.mode') || path.endsWith('.spaceType') || path.endsWith('.quality')) {
            return 'select';
        }
        
        // Most numeric values should be sliders
        if (typeof value === 'number') {
            return 'slider';
        }
        
        // Boolean values should be toggles
        if (typeof value === 'boolean') {
            return 'toggle';
        }
        
        // Color values should use color picker
        if (typeof value === 'string' && value.startsWith('#')) {
            return 'color';
        }
        
        return 'text';
    }

    private createToggleSwitch(path: string, value: boolean): HTMLElement {
        const label = document.createElement('label');
        label.className = 'toggle-switch';
        
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = value;
        
        const slider = document.createElement('span');
        slider.className = 'toggle-slider';
        
        input.addEventListener('change', async () => {
            try {
                await this.settingsStore.set(path, input.checked);
            } catch (error) {
                logger.error(`Failed to update ${path}:`, error);
                input.checked = !input.checked;
            }
        });
        
        label.appendChild(input);
        label.appendChild(slider);
        return label;
    }

    private createSlider(path: string, value: number, valueDisplay: HTMLElement): HTMLElement {
        const container = document.createElement('div');
        container.className = 'slider-container';
        
        const input = document.createElement('input');
        input.type = 'range';
        input.min = '0';
        input.max = '1';
        input.step = getStepValue(path);
        input.value = String(value);
        
        // Set appropriate ranges based on the setting
        if (path.includes('Strength')) {
            input.max = '2';
        } else if (path.includes('Size')) {
            input.max = '10';
        } else if (path.includes('Opacity')) {
            input.max = '1';
        }
        
        valueDisplay.textContent = value.toFixed(2);
        
        input.addEventListener('input', () => {
            valueDisplay.textContent = Number(input.value).toFixed(2);
        });
        
        input.addEventListener('change', async () => {
            try {
                await this.settingsStore.set(path, Number(input.value));
            } catch (error) {
                logger.error(`Failed to update ${path}:`, error);
                input.value = String(value);
                valueDisplay.textContent = value.toFixed(2);
            }
        });
        
        container.appendChild(input);
        return container;
    }

    private createSelect(path: string, value: string): HTMLElement {
        const container = document.createElement('div');
        container.className = 'select-container';
        
        const select = document.createElement('select');
        
        // Add appropriate options based on the setting
        if (path.endsWith('.mode')) {
            ['immersive-ar', 'immersive-vr'].forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = formatSettingName(option);
                select.appendChild(opt);
            });
        } else if (path.endsWith('.spaceType')) {
            ['viewer', 'local', 'local-floor', 'bounded-floor', 'unbounded'].forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = formatSettingName(option);
                select.appendChild(opt);
            });
        } else if (path.endsWith('.quality')) {
            ['low', 'medium', 'high'].forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = formatSettingName(option);
                select.appendChild(opt);
            });
        }
        
        select.value = value;
        
        select.addEventListener('change', async () => {
            try {
                await this.settingsStore.set(path, select.value);
            } catch (error) {
                logger.error(`Failed to update ${path}:`, error);
                select.value = value;
            }
        });
        
        container.appendChild(select);
        return container;
    }

    private createColorPicker(path: string, value: string): HTMLElement {
        const container = document.createElement('div');
        container.className = 'color-picker';
        
        const input = document.createElement('input');
        input.type = 'color';
        input.value = value;
        
        input.addEventListener('change', async () => {
            try {
                await this.settingsStore.set(path, input.value);
            } catch (error) {
                logger.error(`Failed to update ${path}:`, error);
                input.value = value;
            }
        });
        
        container.appendChild(input);
        return container;
    }

    private createTextInput(path: string, value: any): HTMLElement {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = String(value);
        
        input.addEventListener('change', async () => {
            try {
                await this.settingsStore.set(path, input.value);
            } catch (error) {
                logger.error(`Failed to update ${path}:`, error);
                input.value = String(value);
            }
        });
        
        return input;
    }

    private updateInputValue(input: HTMLElement, value: any, valueDisplay?: HTMLElement): void {
        if (input instanceof HTMLInputElement) {
            if (input.type === 'checkbox') {
                input.checked = value as boolean;
            } else if (input.type === 'range') {
                input.value = String(value);
                if (valueDisplay) {
                    valueDisplay.textContent = Number(value).toFixed(2);
                }
            } else {
                input.value = String(value);
            }
        } else if (input instanceof HTMLSelectElement) {
            input.value = String(value);
        }
    }

    public show(): void {
        this.container.classList.add('visible');
    }

    public hide(): void {
        this.container.classList.remove('visible');
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.validationDisplay.dispose();
        this.container.remove();
        ControlPanel.instance = null;
    }
}
