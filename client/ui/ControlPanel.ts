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
            await this.settingsStore.initialize();
            this.settings = this.settingsStore.get('') as Settings;
            this.createPanelElements();
            await this.setupSettingsSubscriptions();
        } catch (error) {
            logger.error('Failed to initialize control panel:', error);
        }
    }

    private createPanelElements(): void {
        // Clear existing content
        this.container.innerHTML = '';

        // Create settings sections
        const flatSettings = this.flattenSettings(this.settings);
        const groupedSettings = this.groupSettingsByCategory(flatSettings);

        for (const [category, settings] of Object.entries(groupedSettings)) {
            const section = this.createSection(category);
            
            for (const [path, value] of Object.entries(settings)) {
                const control = this.createSettingControl(path, value);
                if (control) {
                    section.appendChild(control);
                }
            }
            
            this.container.appendChild(section);
        }
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
        
        for (const [path, value] of Object.entries(flatSettings)) {
            const category = path.split('.')[0];
            if (!result[category]) {
                result[category] = {};
            }
            result[category][path] = value;
        }
        
        return result;
    }

    private createSection(category: string): HTMLElement {
        const section = document.createElement('div');
        section.className = 'settings-section';
        
        const header = document.createElement('h2');
        header.textContent = this.formatCategoryName(category);
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
        return setting
            .split(/(?=[A-Z])/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    private setupSettingsSubscriptions(): void {
        // Clear existing subscriptions
        this.unsubscribers.forEach(unsub => unsub());
        this.unsubscribers = [];

        const settings = this.settingsStore;
        let unsubscriber: (() => void) | undefined;

        // Subscribe to settings changes
        settings.subscribe('visualization.labels.enableLabels', (value) => {
            this.updateLabelVisibility(typeof value === 'boolean' ? value : value === 'true');
        }).then(unsub => {
            unsubscriber = unsub;
            if (unsubscriber) {
                this.unsubscribers.push(unsubscriber);
            }
        });

        const flatSettings = this.flattenSettings(this.settings);
        for (const path of Object.keys(flatSettings)) {
            settings.subscribe(path, (value) => {
                this.updateSettingValue(path, value);
            }).then(unsub => {
                if (unsub) {
                    this.unsubscribers.push(unsub);
                }
            });
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
        const element = document.getElementById(`setting-${path}`);
        if (!element) {
            logger.warn(`No element found for setting: ${path}`);
            return;
        }

        if (element instanceof HTMLInputElement) {
            switch (element.type) {
                case 'checkbox':
                    element.checked = value as boolean;
                    break;
                case 'number':
                    element.value = String(value);
                    break;
                case 'color':
                    element.value = value as string;
                    break;
                default:
                    element.value = String(value);
            }
        } else if (element instanceof HTMLSelectElement) {
            element.value = String(value);
        }
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsub => unsub());
        this.unsubscribers = [];
        this.container.innerHTML = '';
    }
}
