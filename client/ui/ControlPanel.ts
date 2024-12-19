/**
 * Control panel for visualization settings
 */

import { Settings, SettingCategory, SettingKey, SettingValueType } from '../types/settings';
import { settingsManager } from '../state/settings';
import { createLogger } from '../utils/logger';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private currentSettings: Settings;
    private unsubscribers: Array<() => void> = [];

    constructor(container: HTMLElement) {
        this.container = container;
        this.currentSettings = settingsManager.getCurrentSettings();
        this.setupUI();
    }

    private async setupUI(): Promise<void> {
        try {
            Object.entries(this.currentSettings).forEach(([categoryName, settings]) => {
                const category = categoryName as SettingCategory;
                const categoryContainer = document.createElement('div');
                categoryContainer.classList.add('settings-category');
                
                const categoryTitle = document.createElement('h3');
                categoryTitle.textContent = this.formatDisplayName(category);
                categoryContainer.appendChild(categoryTitle);

                Object.entries(settings).forEach(([settingName, value]) => {
                    const setting = settingName as SettingKey<typeof category>;
                    this.createSettingControl(
                        category,
                        setting,
                        value as SettingValueType<typeof category, typeof setting>,
                        categoryContainer
                    );
                });

                this.container.appendChild(categoryContainer);
            });

            const resetButton = document.createElement('button');
            resetButton.textContent = 'Reset to Defaults';
            resetButton.onclick = () => this.resetToDefaults();
            this.container.appendChild(resetButton);

        } catch (error) {
            logger.error('Error setting up UI:', error);
        }
    }

    private formatDisplayName(name: string): string {
        return name
            .replace(/([A-Z])/g, ' $1') // Add space before capital letters
            .replace(/^./, str => str.toUpperCase()); // Capitalize first letter
    }

    private createSettingControl<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>,
        container: HTMLElement
    ): void {
        const controlContainer = document.createElement('div');
        controlContainer.classList.add('setting-control');

        const label = document.createElement('label');
        label.textContent = this.formatDisplayName(String(setting));
        controlContainer.appendChild(label);

        if (Array.isArray(value)) {
            // Handle array values (e.g., widthRange, sizeRange)
            const arrayContainer = document.createElement('div');
            arrayContainer.classList.add('array-control');
            
            value.forEach((item, index) => {
                const itemInput = document.createElement('input');
                itemInput.type = 'number';
                itemInput.value = String(item);
                itemInput.step = '0.1';
                itemInput.onchange = async (event) => {
                    const target = event.target as HTMLInputElement;
                    const newArray = [...value];
                    newArray[index] = parseFloat(target.value);
                    try {
                        await settingsManager.updateSetting(category, setting, newArray as SettingValueType<T, K>);
                    } catch (error) {
                        logger.error(`Error updating array setting ${category}.${String(setting)}:`, error);
                        target.value = String(item);
                    }
                };
                arrayContainer.appendChild(itemInput);
            });
            
            controlContainer.appendChild(arrayContainer);
        } else if (typeof value === 'object' && value !== null) {
            // Handle nested objects
            logger.warn(`Nested object setting not supported in UI: ${category}.${String(setting)}`);
            return;
        } else {
            let input: HTMLInputElement;
            
            switch (typeof value) {
                case 'boolean':
                    input = document.createElement('input');
                    input.type = 'checkbox';
                    input.checked = value as boolean;
                    break;

                case 'number':
                    input = document.createElement('input');
                    input.type = 'number';
                    input.value = String(value);
                    input.step = '0.1';
                    break;

                case 'string':
                    input = document.createElement('input');
                    if (value.startsWith('#')) {
                        input.type = 'color';
                    } else {
                        input.type = 'text';
                    }
                    input.value = value;
                    break;

                default:
                    logger.warn(`Unsupported setting type for ${category}.${String(setting)}`);
                    return;
            }

            input.onchange = async (event) => {
                const target = event.target as HTMLInputElement;
                let newValue: SettingValueType<T, K>;

                switch (target.type) {
                    case 'checkbox':
                        newValue = target.checked as SettingValueType<T, K>;
                        break;
                    case 'number':
                        newValue = parseFloat(target.value) as SettingValueType<T, K>;
                        break;
                    default:
                        newValue = target.value as SettingValueType<T, K>;
                }

                try {
                    await settingsManager.updateSetting(category, setting, newValue);
                } catch (error) {
                    logger.error(`Error updating setting ${category}.${String(setting)}:`, error);
                    // Revert the UI to the previous value
                    if (target.type === 'checkbox') {
                        target.checked = value as boolean;
                    } else {
                        target.value = String(value);
                    }
                }
            };

            controlContainer.appendChild(input);
        }

        container.appendChild(controlContainer);
    }

    private async resetToDefaults(): Promise<void> {
        try {
            const defaultSettings = settingsManager.getDefaultSettings();
            for (const [categoryName, settings] of Object.entries(defaultSettings)) {
                const category = categoryName as SettingCategory;
                for (const [settingName, value] of Object.entries(settings)) {
                    const setting = settingName as SettingKey<typeof category>;
                    await settingsManager.updateSetting(
                        category,
                        setting,
                        value as SettingValueType<typeof category, typeof setting>
                    );
                }
            }
        } catch (error) {
            logger.error('Error resetting to defaults:', error);
        }
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
