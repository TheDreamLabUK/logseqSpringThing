/**
 * Control panel for visualization settings
 */

import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';
import { Settings, SettingKey, SettingValue } from '../core/types';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private unsubscribers: Array<() => void> = [];

    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'control-panel';
        this.initializeUI();
    }

    private initializeUI(): void {
        // Create header
        const header = document.createElement('div');
        header.className = 'control-panel-header';
        
        const title = document.createElement('h2');
        title.textContent = 'Settings';
        
        header.appendChild(title);
        this.container.appendChild(header);

        // Create settings sections
        const settings = settingsManager.getCurrentSettings();
        Object.entries(settings).forEach(([category, categorySettings]) => {
            this.createCategorySection(category as keyof Settings, categorySettings);
        });

        // Create reset button
        const resetButton = document.createElement('button');
        resetButton.className = 'reset-button';
        resetButton.textContent = 'Reset to Defaults';
        resetButton.onclick = () => this.resetToDefaults();
        this.container.appendChild(resetButton);
    }

    private createCategorySection(category: keyof Settings, settings: any): void {
        const section = document.createElement('div');
        section.className = 'settings-section';
        
        // Convert category from camelCase to Title Case for display
        const title = document.createElement('h3');
        title.textContent = String(category)
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, (str: string) => str.toUpperCase());
        section.appendChild(title);

        Object.entries(settings).forEach(([setting, value]) => {
            const settingElement = this.createSettingElement(
                category,
                setting as SettingKey<typeof category>,
                value as SettingValue
            );
            section.appendChild(settingElement);
        });

        this.container.appendChild(section);
    }

    private createSettingElement(category: keyof Settings, setting: SettingKey<typeof category>, value: SettingValue): HTMLElement {
        const container = document.createElement('div');
        container.className = 'setting-container';

        // Convert setting from camelCase to Title Case for display
        const label = document.createElement('label');
        const settingStr = String(setting); // Convert to string to ensure replace method exists
        label.textContent = settingStr
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, (str: string) => str.toUpperCase());

        let input: HTMLElement;

        if (typeof value === 'boolean') {
            input = this.createInputElement('checkbox', value, (e) => {
                const target = e.target as HTMLInputElement;
                settingsManager.updateSetting(category, setting, target.checked);
            });
        } else if (typeof value === 'number') {
            input = this.createInputElement('number', value, (e) => {
                const target = e.target as HTMLInputElement;
                settingsManager.updateSetting(category, setting, parseFloat(target.value));
            });
        } else if (typeof value === 'string') {
            if (value.startsWith('#')) {
                input = this.createInputElement('color', value, (e) => {
                    const target = e.target as HTMLInputElement;
                    settingsManager.updateSetting(category, setting, target.value);
                });
            } else {
                input = this.createInputElement('text', value, (e) => {
                    const target = e.target as HTMLInputElement;
                    settingsManager.updateSetting(category, setting, target.value);
                });
            }
        } else if (Array.isArray(value)) {
            const arrayContainer = document.createElement('div');
            arrayContainer.className = 'array-input';
            value.forEach((item, index) => {
                const itemInput = this.createInputElement('number', item, (e) => {
                    const target = e.target as HTMLInputElement;
                    const newValue = [...value];
                    newValue[index] = parseFloat(target.value);
                    settingsManager.updateSetting(category, setting, newValue);
                });
                arrayContainer.appendChild(itemInput);
            });
            input = arrayContainer;
        } else {
            input = document.createElement('div');
            input.textContent = 'Unsupported type';
        }

        container.appendChild(label);
        container.appendChild(input);

        // Subscribe to changes
        this.unsubscribers.push(
            settingsManager.subscribe(category, setting, (newValue) => {
                if (input instanceof HTMLInputElement) {
                    if (input.type === 'checkbox') {
                        input.checked = newValue as boolean;
                    } else {
                        input.value = String(newValue);
                    }
                } else if (input.className === 'array-input' && Array.isArray(newValue)) {
                    const inputs = input.getElementsByTagName('input');
                    for (let i = 0; i < inputs.length; i++) {
                        inputs[i].value = String(newValue[i]);
                    }
                }
                this.updateUI();
            })
        );

        return container;
    }

    private createInputElement(
        type: string,
        value: string | number | boolean,
        onChange: (e: Event) => void
    ): HTMLInputElement {
        const input = document.createElement('input');
        input.type = type;
        if (type === 'checkbox') {
            input.checked = value as boolean;
        } else {
            input.value = String(value);
            if (type === 'number') {
                input.step = '0.1';
            }
        }
        input.onchange = onChange;
        return input;
    }

    private updateUI(): void {
        // Update all UI elements that depend on settings
        const settings = settingsManager.getCurrentSettings();
        
        // Update theme
        document.body.style.backgroundColor = settings.rendering.backgroundColor;
        
        // Update control panel visibility
        const panel = document.getElementById('control-panel');
        if (panel) {
            panel.style.display = settings.clientDebug.enabled ? 'block' : 'none';
        }

        // Update other UI elements based on settings
        this.updateLabels(settings);
        this.updateEdges(settings);
        this.updateNodes(settings);
    }

    private updateLabels(settings: Settings): void {
        const labelElements = document.querySelectorAll('.node-label');
        labelElements.forEach(label => {
            if (label instanceof HTMLElement) {
                label.style.display = settings.labels.enableLabels ? 'block' : 'none';
                label.style.fontSize = `${settings.labels.desktopFontSize}px`;
                label.style.color = settings.labels.textColor;
            }
        });
    }

    private updateEdges(settings: Settings): void {
        const edgeElements = document.querySelectorAll('.edge');
        edgeElements.forEach(edge => {
            if (edge instanceof HTMLElement) {
                edge.style.opacity = settings.edges.opacity.toString();
                edge.style.stroke = settings.edges.color;
                edge.style.strokeWidth = settings.edges.baseWidth.toString();
            }
        });
    }

    private updateNodes(settings: Settings): void {
        const nodeElements = document.querySelectorAll('.node');
        nodeElements.forEach(node => {
            if (node instanceof HTMLElement) {
                node.style.backgroundColor = settings.nodes.baseColor;
                node.style.opacity = settings.nodes.opacity.toString();
                const scale = settings.nodes.baseSize;
                node.style.transform = `scale(${scale})`;
            }
        });
    }

    private async resetToDefaults(): Promise<void> {
        try {
            const defaultSettings = settingsManager.getCurrentSettings();
            await settingsManager.updateAllSettings(defaultSettings);
            logger.info('Settings reset to defaults');
            this.updateUI();
        } catch (error) {
            logger.error('Failed to reset settings:', error);
        }
    }

    public mount(parent: HTMLElement): void {
        parent.appendChild(this.container);
    }

    public unmount(): void {
        this.container.remove();
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
