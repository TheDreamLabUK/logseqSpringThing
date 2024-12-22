import { Settings } from '../core/types';
import { settingsManager } from '../state/settings';
import { defaultSettings } from '../state/defaultSettings';
import { createLogger } from '../utils/logger';

const logger = createLogger('ControlPanel');

type SettingsKey<T extends keyof Settings> = keyof Settings[T];
type SettingValue<T extends keyof Settings, K extends SettingsKey<T>> = Settings[T][K];

export class ControlPanel {
    private container: HTMLElement;
    private currentSettings: Settings;
    private unsubscribers: Array<() => void> = [];

    constructor(container: HTMLElement) {
        this.container = container;
        // Start with default settings
        this.currentSettings = { ...defaultSettings };
        this.setupUI();
        this.setupWebSocketStatus();
        this.initializeSettings();
    }

    private async initializeSettings(): Promise<void> {
        try {
            // Subscribe to settings changes
            Object.keys(this.currentSettings).forEach(category => {
                const categoryKey = category as keyof Settings;
                const settings = this.currentSettings[categoryKey];
                Object.keys(settings).forEach(setting => {
                    const settingKey = setting as SettingsKey<typeof categoryKey>;
                    const unsubscribe = settingsManager.subscribe(
                        categoryKey,
                        settingKey,
                        (value: SettingValue<typeof categoryKey, typeof settingKey>) => {
                            this.updateSettingUI(categoryKey, settingKey, value);
                        }
                    );
                    this.unsubscribers.push(unsubscribe);
                });
            });

            // Initialize settings manager
            await settingsManager.initialize();
            // Update UI with current settings
            this.currentSettings = settingsManager.getCurrentSettings();
            this.updateAllSettings();
        } catch (error) {
            logger.error('Error initializing settings:', error);
        }
    }

    private updateSettingUI<T extends keyof Settings, K extends SettingsKey<T>>(
        category: T,
        setting: K,
        value: SettingValue<T, K>
    ): void {
        const categoryEl = this.container.querySelector(`.settings-group[data-category="${category}"]`);
        if (!categoryEl) return;

        const settingEl = categoryEl.querySelector(`.setting-item[data-setting="${String(setting)}"]`);
        if (!settingEl) return;

        const input = settingEl.querySelector('input');
        if (!input) return;

        if (input.type === 'checkbox' && typeof value === 'boolean') {
            (input as HTMLInputElement).checked = value;
        } else {
            input.value = String(value);
        }
    }

    private updateAllSettings(): void {
        Object.entries(this.currentSettings).forEach(([category, settings]) => {
            const categoryKey = category as keyof Settings;
            Object.entries(settings).forEach(([setting, value]) => {
                const settingKey = setting as SettingsKey<typeof categoryKey>;
                this.updateSettingUI(
                    categoryKey,
                    settingKey,
                    value as SettingValue<typeof categoryKey, typeof settingKey>
                );
            });
        });
    }

    private async setupUI(): Promise<void> {
        try {
            // Clear any existing content
            this.container.innerHTML = '';
            
            // Add control panel container
            this.container.classList.add('control-panel');

            // Add header
            const header = document.createElement('div');
            header.classList.add('control-panel-header');
            header.innerHTML = `
                <h3>Settings</h3>
                <div class="connection-status disconnected">
                    <span id="connection-status">Disconnected</span>
                </div>
            `;
            this.container.appendChild(header);

            // Add content container
            const content = document.createElement('div');
            content.classList.add('control-panel-content');
            this.container.appendChild(content);

            // Add settings categories
            (Object.keys(this.currentSettings) as Array<keyof Settings>).forEach(category => {
                const categorySettings = this.currentSettings[category];
                const categoryElement = this.createCategoryElement(category, categorySettings);
                content.appendChild(categoryElement);
            });
        } catch (error) {
            logger.error('Error setting up UI:', error);
        }
    }

    private createCategoryElement<T extends keyof Settings>(
        category: T,
        settings: Settings[T]
    ): HTMLElement {
        const element = document.createElement('div');
        element.classList.add('settings-group');
        element.dataset.category = category;
        
        const title = this.formatTitle(category);
        element.innerHTML = `<h4>${title}</h4>`;

        Object.entries(settings).forEach(([key, value]) => {
            const settingKey = key as SettingsKey<T>;
            const control = this.createSettingControl(category, settingKey, value);
            element.appendChild(control);
        });

        return element;
    }

    private formatTitle(str: string): string {
        return str.replace(/([A-Z])/g, ' $1').trim();
    }

    private createSettingControl<T extends keyof Settings, K extends SettingsKey<T>>(
        category: T,
        key: K,
        value: SettingValue<T, K>
    ): HTMLElement {
        const control = document.createElement('div');
        control.classList.add('setting-item');
        control.dataset.setting = String(key);

        const label = document.createElement('label');
        label.textContent = this.formatTitle(String(key));
        control.appendChild(label);

        if (Array.isArray(value)) {
            const arrayControl = document.createElement('div');
            arrayControl.classList.add('array-inputs');
            value.forEach((item, index) => {
                const input = document.createElement('input');
                input.type = typeof item === 'number' ? 'number' : 'text';
                input.value = String(item);
                input.dataset.index = index.toString();

                // Update settings on input change
                input.addEventListener('input', async () => {
                    const newValue = input.type === 'number' ? parseFloat(input.value) : input.value;
                    const newArray = [...value];
                    newArray[index] = newValue;
                    await this.updateSetting(category, key, newArray as SettingValue<T, K>);
                });

                arrayControl.appendChild(input);
            });
            control.appendChild(arrayControl);
        } else {
            const input = document.createElement('input');
            switch (typeof value) {
                case 'boolean':
                    input.type = 'checkbox';
                    input.checked = value;
                    break;
                case 'number':
                    input.type = 'number';
                    input.value = String(value);
                    break;
                default:
                    input.type = 'text';
                    input.value = String(value);
            }
            control.appendChild(input);

            // Update settings on input change
            input.addEventListener('input', async () => {
                let newValue: SettingValue<T, K>;
                if (input.type === 'checkbox') {
                    newValue = input.checked as SettingValue<T, K>;
                } else if (input.type === 'number') {
                    newValue = parseFloat(input.value) as SettingValue<T, K>;
                } else {
                    newValue = input.value as SettingValue<T, K>;
                }
                await this.updateSetting(category, key, newValue);
            });
        }

        return control;
    }

    private async updateSetting<T extends keyof Settings, K extends SettingsKey<T>>(
        category: T,
        key: K,
        value: SettingValue<T, K>
    ): Promise<void> {
        try {
            // Update local settings
            if (this.currentSettings[category]) {
                (this.currentSettings[category] as any)[key] = value;
            }
            // Update server settings
            await settingsManager.updateSetting(
                category,
                key,
                value
            );
        } catch (error) {
            logger.error('Error updating setting:', error);
            // Revert UI to current setting value
            this.updateSettingUI(category, key, (this.currentSettings[category] as any)[key]);
        }
    }

    private setupWebSocketStatus(): void {
        const statusIndicator = this.container.querySelector('.connection-status');
        const statusText = this.container.querySelector('#connection-status');
        
        if (statusIndicator && statusText) {
            try {
                // Use the default WebSocket URL
                const wsUrl = new URL('/wss', window.location.href);
                wsUrl.protocol = wsUrl.protocol.replace('http', 'ws');
                const ws = new WebSocket(wsUrl.toString());
                
                ws.onopen = () => {
                    statusIndicator.classList.remove('disconnected');
                    statusText.textContent = 'Connected';
                };
                
                ws.onclose = () => {
                    statusIndicator.classList.add('disconnected');
                    statusText.textContent = 'Disconnected';
                };
                
                ws.onerror = () => {
                    statusIndicator.classList.add('disconnected');
                    statusText.textContent = 'Error';
                };
            } catch (error) {
                logger.error('Error setting up WebSocket:', error);
            }
        }
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
