/**
 * Control panel for visualization settings
 */

import { Settings, SettingsCategory, SettingKey, SettingValueType } from '../types/settings';
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
                const category = categoryName as SettingsCategory;
                const categoryContainer = document.createElement('div');
                categoryContainer.classList.add('settings-category');
                
                const categoryTitle = document.createElement('h3');
                categoryTitle.textContent = category;
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

    private createSettingControl<T extends SettingsCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>,
        container: HTMLElement
    ): void {
        const controlContainer = document.createElement('div');
        controlContainer.classList.add('setting-control');

        const label = document.createElement('label');
        label.textContent = String(setting);
        controlContainer.appendChild(label);

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
        container.appendChild(controlContainer);
    }

    private async resetToDefaults(): Promise<void> {
        try {
            const defaultSettings = settingsManager.getDefaultSettings();
            for (const [categoryName, settings] of Object.entries(defaultSettings)) {
                const category = categoryName as SettingsCategory;
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
