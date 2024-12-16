/**
 * Control panel for visualization settings
 */

import { createLogger } from '../utils/logger';
import { settingsManager, Settings } from '../state/settings';
import './ControlPanel.css';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private settings: Settings;
    private unsubscribers: (() => void)[] = [];
    private panel: HTMLElement | null;
    private expanded: boolean = false;

    constructor() {
        this.settings = settingsManager.getCurrentSettings();
        this.panel = document.getElementById('settings-panel');
        
        if (!this.panel) {
            logger.error('Settings panel element not found');
            return;
        }

        this.setupToggleButton();
        this.setupSubscriptions();
        this.setupUI();
        this.setupEventListeners();
    }

    private setupToggleButton() {
        const toggleButton = this.panel?.querySelector('.toggle-button');
        if (toggleButton) {
            toggleButton.addEventListener('click', () => this.togglePanel());
        }
    }

    private togglePanel() {
        this.expanded = !this.expanded;
        if (this.panel) {
            this.panel.classList.toggle('expanded', this.expanded);
        }
    }

    private setupSubscriptions() {
        // Subscribe to connection status
        const unsubscribeConnection = settingsManager.subscribeToConnection(this.updateConnectionStatus);
        this.unsubscribers.push(unsubscribeConnection);

        // Subscribe to all settings
        Object.keys(this.settings).forEach(category => {
            const categorySettings = this.settings[category as keyof Settings];
            Object.keys(categorySettings).forEach(setting => {
                const unsubscribe = settingsManager.subscribe(category, setting, (value: any) => {
                    (this.settings[category as keyof Settings] as any)[setting] = value;
                    this.updateUI(category, setting);
                });
                this.unsubscribers.push(unsubscribe);
            });
        });
    }

    private updateConnectionStatus = (connected: boolean) => {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = connected ? 'connected' : 'disconnected';
        }
    };

    private async updateSetting(category: keyof Settings, setting: string, value: any) {
        try {
            const updateMethod = `update${category.charAt(0).toUpperCase() + category.slice(1)}Setting`;
            await (settingsManager as any)[updateMethod](setting, value);
            this.showFeedback('success', `Updated ${category}.${setting}`);
        } catch (error) {
            logger.error(`Failed to update ${String(category)}.${setting}:`, error);
            this.showFeedback('error', `Failed to update ${category}.${setting}`);
        }
    }

    private showFeedback(type: 'success' | 'error', message: string) {
        const feedback = document.createElement('div');
        feedback.className = `settings-feedback ${type}`;
        feedback.textContent = message;
        
        const content = this.panel?.querySelector('.control-panel-content');
        if (content) {
            content.appendChild(feedback);
            setTimeout(() => {
                feedback.classList.add('fade-out');
                setTimeout(() => feedback.remove(), 300);
            }, 2000);
        }
    }

    private updateUI(category: string, setting: string) {
        const element = document.getElementById(`${category}-${setting}`);
        if (element) {
            const value = (this.settings[category as keyof Settings] as any)[setting];
            if (element instanceof HTMLInputElement) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else if (element.type === 'color') {
                    element.value = value;
                } else if (element.type === 'number') {
                    element.value = value.toString();
                } else {
                    element.value = value;
                }
            } else if (element instanceof HTMLSelectElement) {
                element.value = value;
            }
        }
    }

    private setupUI() {
        const content = this.panel?.querySelector('.control-panel-content');
        if (!content) return;

        // Clear existing content
        content.innerHTML = '';

        // Create sections for each category
        Object.entries(this.settings).forEach(([category, settings]) => {
            const section = this.createSection(category, settings);
            content.appendChild(section);
        });
    }

    private createSection(category: string, settings: any): HTMLElement {
        const section = document.createElement('div');
        section.className = 'settings-group';
        
        const header = document.createElement('h4');
        header.textContent = this.formatCategoryName(category);
        section.appendChild(header);

        Object.entries(settings).forEach(([setting, value]) => {
            const item = this.createSettingItem(category, setting, value);
            section.appendChild(item);
        });

        return section;
    }

    private createSettingItem(category: string, setting: string, value: any): HTMLElement {
        const item = document.createElement('div');
        item.className = 'setting-item';

        const label = document.createElement('label');
        label.textContent = this.formatSettingName(setting);
        label.htmlFor = `${category}-${setting}`;

        const input = this.createInput(category, setting, value);
        input.id = `${category}-${setting}`;

        item.appendChild(label);
        item.appendChild(input);

        return item;
    }

    private createInput(category: string, setting: string, value: any): HTMLElement {
        if (typeof value === 'boolean') {
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value;
            input.addEventListener('change', (e) => {
                const target = e.target as HTMLInputElement;
                this.updateSetting(category as keyof Settings, setting, target.checked);
            });
            return input;
        }

        if (typeof value === 'number') {
            const input = document.createElement('input');
            input.type = 'number';
            input.value = value.toString();
            input.step = setting.includes('opacity') ? '0.1' : '1';
            input.addEventListener('change', (e) => {
                const target = e.target as HTMLInputElement;
                this.updateSetting(category as keyof Settings, setting, parseFloat(target.value));
            });
            return input;
        }

        if (typeof value === 'string') {
            if (setting.toLowerCase().includes('color')) {
                const input = document.createElement('input');
                input.type = 'color';
                input.value = value;
                input.addEventListener('change', (e) => {
                    const target = e.target as HTMLInputElement;
                    this.updateSetting(category as keyof Settings, setting, target.value);
                });
                return input;
            }

            const input = document.createElement('input');
            input.type = 'text';
            input.value = value;
            input.addEventListener('change', (e) => {
                const target = e.target as HTMLInputElement;
                this.updateSetting(category as keyof Settings, setting, target.value);
            });
            return input;
        }

        // For arrays or other types, create a disabled text input
        const input = document.createElement('input');
        input.type = 'text';
        input.value = JSON.stringify(value);
        input.disabled = true;
        return input;
    }

    private setupEventListeners() {
        const resetButton = document.getElementById('reset-settings');
        if (resetButton) {
            resetButton.addEventListener('click', async () => {
                try {
                    this.settings = settingsManager.getCurrentSettings();
                    await settingsManager.loadAllSettings();
                    this.showFeedback('success', 'Settings reset to defaults');
                } catch (error) {
                    logger.error('Failed to reset settings:', error);
                    this.showFeedback('error', 'Failed to reset settings');
                }
            });
        }

        const saveButton = document.getElementById('save-settings');
        if (saveButton) {
            saveButton.addEventListener('click', async () => {
                try {
                    await settingsManager.loadAllSettings();
                    this.showFeedback('success', 'Settings saved');
                } catch (error) {
                    logger.error('Failed to save settings:', error);
                    this.showFeedback('error', 'Failed to save settings');
                }
            });
        }
    }

    private formatCategoryName(name: string): string {
        return name.charAt(0).toUpperCase() + name.slice(1) + ' Settings';
    }

    private formatSettingName(name: string): string {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    dispose() {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
