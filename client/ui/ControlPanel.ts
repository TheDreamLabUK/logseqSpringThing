import { Settings } from '../core/types';
import { settingsManager } from '../state/settings';
import { platformManager } from '../platform/platformManager';
import { createLogger } from '../core/logger';
import './ControlPanel.css';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private settings: Settings;
    private unsubscribers: Array<() => void> = [];

    constructor(container: HTMLElement) {
        this.container = container;
        this.settings = settingsManager.getCurrentSettings();
        this.initializePanel();
        this.setupSettingsSubscriptions();
    }

    private initializePanel(): void {
        // Create settings sections
        const categories = Object.keys(this.settings) as Array<keyof Settings>;
        categories.forEach(category => {
            const section = this.createSettingsSection(category);
            this.container.appendChild(section);
        });

        // Add platform-specific settings
        if (platformManager.getCapabilities().xrSupported) {
            this.addXRSettings();
        }
    }

    private createSettingsSection(category: keyof Settings): HTMLElement {
        const section = document.createElement('div');
        section.className = 'settings-group';
        
        const title = document.createElement('h4');
        title.textContent = this.formatCategoryName(String(category));
        section.appendChild(title);

        const settings = this.settings[category];
        Object.entries(settings).forEach(([key, value]) => {
            const settingElement = this.createSettingElement(
                category,
                key as keyof Settings[typeof category],
                value as Settings[typeof category][keyof Settings[typeof category]]
            );
            section.appendChild(settingElement);
        });

        return section;
    }

    private formatCategoryName(category: string): string {
        return category
            .split(/(?=[A-Z])/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    private createSettingElement<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        value: Settings[T][K]
    ): HTMLElement {
        const container = document.createElement('div');
        container.className = 'setting-item';

        const label = document.createElement('label');
        label.textContent = this.formatSettingName(String(key));
        container.appendChild(label);

        const input = this.createInputElement(category, key, value);
        container.appendChild(input);

        return container;
    }

    private formatSettingName(setting: string): string {
        return setting
            .split(/(?=[A-Z])|_/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    private createInputElement<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        value: Settings[T][K]
    ): HTMLElement {
        let input: HTMLElement;

        switch (typeof value) {
            case 'boolean':
                input = document.createElement('input');
                input.setAttribute('type', 'checkbox');
                (input as HTMLInputElement).checked = value as boolean;
                break;

            case 'number':
                input = document.createElement('input');
                input.setAttribute('type', 'number');
                input.setAttribute('step', '0.1');
                (input as HTMLInputElement).value = String(value);
                break;

            case 'string':
                if (this.isColorSetting(String(key))) {
                    input = document.createElement('input');
                    input.setAttribute('type', 'color');
                    (input as HTMLInputElement).value = value as string;
                } else {
                    input = document.createElement('input');
                    input.setAttribute('type', 'text');
                    (input as HTMLInputElement).value = value as string;
                }
                break;

            default:
                logger.warn(`Unsupported setting type for ${String(key)}: ${typeof value}`);
                input = document.createElement('span');
                input.textContent = 'Unsupported setting type';
                break;
        }

        input.id = `${String(category)}-${String(key)}`;
        input.addEventListener('change', (event) => this.handleSettingChange(category, key, event));

        return input;
    }

    private isColorSetting(key: string): boolean {
        return key.toLowerCase().includes('color');
    }

    private async handleSettingChange<T extends keyof Settings, K extends keyof Settings[T]>(
        category: T,
        key: K,
        event: Event
    ): Promise<void> {
        const target = event.target as HTMLInputElement;
        let value: Settings[T][K];

        switch (target.type) {
            case 'checkbox':
                value = target.checked as Settings[T][K];
                break;
            case 'number':
                value = parseFloat(target.value) as Settings[T][K];
                break;
            default:
                value = target.value as Settings[T][K];
        }

        try {
            await settingsManager.updateSetting(category, key, value);
            logger.info(`Updated setting ${String(category)}.${String(key)} to ${value}`);
        } catch (error) {
            logger.error(`Failed to update setting ${String(category)}.${String(key)}:`, error);
            // Revert input to current setting value
            const currentValue = this.settings[category][key];
            if (target.type === 'checkbox') {
                target.checked = currentValue as boolean;
            } else {
                target.value = String(currentValue);
            }
        }
    }

    private setupSettingsSubscriptions(): void {
        // Subscribe to all settings
        Object.keys(this.settings).forEach(category => {
            const categoryKey = category as keyof Settings;
            const categorySettings = this.settings[categoryKey];
            Object.keys(categorySettings).forEach(setting => {
                const settingKey = setting as keyof Settings[typeof categoryKey];
                const unsubscribe = settingsManager.subscribe(
                    categoryKey,
                    settingKey,
                    (value: Settings[typeof categoryKey][typeof settingKey]) => {
                        this.updateSettingElement(category, setting, value);
                    }
                );
                this.unsubscribers.push(unsubscribe);
            });
        });
    }

    private updateSettingElement(category: string, setting: string, value: unknown): void {
        const input = document.getElementById(`${category}-${setting}`) as HTMLInputElement;
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = value as boolean;
            } else {
                input.value = String(value);
            }
        }
    }

    private addXRSettings(): void {
        // Add XR-specific settings section
        const xrSection = document.createElement('div');
        xrSection.className = 'settings-group';
        
        const title = document.createElement('h4');
        title.textContent = 'XR Settings';
        xrSection.appendChild(title);

        // Add XR mode toggle
        const xrToggle = document.createElement('button');
        xrToggle.id = 'xr-toggle';
        xrToggle.textContent = 'Enter XR';
        xrToggle.addEventListener('click', () => {
            // XR mode toggle logic handled by XRManager
            const event = new CustomEvent('toggleXR');
            window.dispatchEvent(event);
        });
        xrSection.appendChild(xrToggle);

        this.container.appendChild(xrSection);
    }

    public dispose(): void {
        // Clean up subscribers
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
