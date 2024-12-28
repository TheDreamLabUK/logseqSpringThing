import { ErrorDisplay } from './ErrorDisplay';
import { Setting } from '../state/SettingsStore';
import { Logger } from '../utils/logger';

const log = new Logger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private errorDisplay: ErrorDisplay;
    private settingsElements: Map<string, HTMLElement> = new Map();

    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'control-panel';
        this.errorDisplay = new ErrorDisplay();
    }

    public createSettingControl(path: string, setting: Setting): HTMLElement {
        const container = document.createElement('div');
        container.className = 'setting-control';

        const label = document.createElement('label');
        label.textContent = setting.description || path;

        let input: HTMLElement;

        try {
            switch (setting.type) {
                case 'number':
                    input = this.createNumberInput(setting);
                    break;
                case 'string':
                    input = setting.options ? 
                        this.createSelectInput(setting) :
                        this.createTextInput(setting);
                    break;
                case 'boolean':
                    input = this.createBooleanInput(setting);
                    break;
                case 'array':
                    input = this.createArrayInput(setting);
                    break;
                case 'object':
                    input = this.createObjectInput(setting);
                    break;
                default:
                    throw new Error(`Unsupported setting type: ${setting.type}`);
            }
        } catch (error) {
            log.error(`Failed to create control for setting ${path}:`, error);
            input = document.createElement('div');
            input.className = 'error';
            input.textContent = 'Error: Failed to create control';
        }

        container.appendChild(label);
        container.appendChild(input);
        this.settingsElements.set(path, input);

        return container;
    }

    private createNumberInput(setting: Setting): HTMLInputElement {
        const input = document.createElement('input');
        input.type = 'number';
        input.value = setting.value;
        if (setting.min !== undefined) input.min = setting.min.toString();
        if (setting.max !== undefined) input.max = setting.max.toString();
        return input;
    }

    private createSelectInput(setting: Setting): HTMLSelectElement {
        const select = document.createElement('select');
        setting.options?.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            optionElement.selected = option === setting.value;
            select.appendChild(optionElement);
        });
        return select;
    }

    private createTextInput(setting: Setting): HTMLInputElement {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = setting.value;
        return input;
    }

    private createBooleanInput(setting: Setting): HTMLInputElement {
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = setting.value;
        return input;
    }

    private createArrayInput(setting: Setting): HTMLElement {
        const container = document.createElement('div');
        container.className = 'array-input';
        
        setting.value.forEach((item: any, index: number) => {
            const itemContainer = document.createElement('div');
            const itemInput = this.createSettingControl(`${setting}.${index}`, {
                value: item,
                type: typeof item
            });
            itemContainer.appendChild(itemInput);
            container.appendChild(itemContainer);
        });

        const addButton = document.createElement('button');
        addButton.textContent = 'Add Item';
        addButton.onclick = () => {
            // Add new item logic
        };
        container.appendChild(addButton);

        return container;
    }

    private createObjectInput(setting: Setting): HTMLElement {
        const container = document.createElement('div');
        container.className = 'object-input';
        
        Object.entries(setting.value).forEach(([key, value]) => {
            const propertyContainer = document.createElement('div');
            const propertyInput = this.createSettingControl(`${setting}.${key}`, {
                value: value,
                type: typeof value
            });
            propertyContainer.appendChild(propertyInput);
            container.appendChild(propertyContainer);
        });

        return container;
    }

    public async saveSettings(): Promise<void> {
        try {
            const updates = Array.from(this.settingsElements.entries())
                .map(([path, element]) => ({
                    path,
                    value: this.getElementValue(element)
                }));

            await this.settingsStore.batchUpdate(updates);
            this.errorDisplay.show('Settings saved successfully', 3000);
        } catch (error) {
            log.error('Failed to save settings:', error);
            this.errorDisplay.show(`Failed to save settings: ${error.message}`);
        }
    }

    private getElementValue(element: HTMLElement): any {
        if (element instanceof HTMLInputElement) {
            if (element.type === 'checkbox') {
                return element.checked;
            }
            if (element.type === 'number') {
                return Number(element.value);
            }
            return element.value;
        }
        if (element instanceof HTMLSelectElement) {
            return element.value;
        }
        // Handle array and object inputs
        if (element.classList.contains('array-input')) {
            return Array.from(element.children)
                .filter(child => child instanceof HTMLElement)
                .map(child => this.getElementValue(child as HTMLElement));
        }
        if (element.classList.contains('object-input')) {
            const obj: Record<string, any> = {};
            Array.from(element.children)
                .filter(child => child instanceof HTMLElement)
                .forEach(child => {
                    const key = child.getAttribute('data-key');
                    if (key) {
                        obj[key] = this.getElementValue(child as HTMLElement);
                    }
                });
            return obj;
        }
        return null;
    }

    public dispose(): void {
        this.errorDisplay.dispose();
        this.container.remove();
    }
} 