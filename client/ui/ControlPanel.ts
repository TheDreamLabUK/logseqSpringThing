/**
 * Control panel for visualization settings
 */

import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';
import { Settings, NodeSettings, EdgeSettings, PhysicsSettings, RenderingSettings, BloomSettings, AnimationSettings, LabelSettings, ARSettings } from '../state/settings';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private settings: Settings;
    private unsubscribers: (() => void)[] = [];

    constructor() {
        this.settings = settingsManager.getCurrentSettings();
        this.setupSubscriptions();
        this.setupUI();
    }

    private setupSubscriptions() {
        // Subscribe to individual setting changes
        const categories = ['nodes', 'edges', 'physics', 'rendering', 'bloom', 'animation', 'label', 'ar'] as const;
        
        categories.forEach(category => {
            Object.keys(this.settings[category]).forEach(setting => {
                const unsubscribe = settingsManager.subscribe(category, setting, (value: any) => {
                    (this.settings[category] as any)[setting] = value;
                    this.updateUI(category, setting);
                });
                this.unsubscribers.push(unsubscribe);
            });
        });

        // Subscribe to connection status
        const unsubscribeConnection = settingsManager.subscribeToConnection(this.updateConnectionStatus);
        this.unsubscribers.push(unsubscribeConnection);
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
            switch (category) {
                case 'nodes':
                    await settingsManager.updateNodeSetting(setting as keyof NodeSettings, value);
                    break;
                case 'edges':
                    await settingsManager.updateEdgeSetting(setting as keyof EdgeSettings, value);
                    break;
                case 'physics':
                    await settingsManager.updatePhysicsSetting(setting as keyof PhysicsSettings, value);
                    break;
                case 'rendering':
                    await settingsManager.updateRenderingSetting(setting as keyof RenderingSettings, value);
                    break;
                case 'bloom':
                    await settingsManager.updateBloomSetting(setting as keyof BloomSettings, value);
                    break;
                case 'animation':
                    await settingsManager.updateAnimationSetting(setting as keyof AnimationSettings, value);
                    break;
                case 'label':
                    await settingsManager.updateLabelSetting(setting as keyof LabelSettings, value);
                    break;
                case 'ar':
                    await settingsManager.updateARSetting(setting as keyof ARSettings, value);
                    break;
            }
        } catch (error) {
            logger.error(`Failed to update ${String(category)}.${setting}:`, error);
        }
    }

    private updateUI(category: string, setting: string) {
        const element = document.getElementById(`${category}-${setting}`);
        if (element) {
            const value = (this.settings[category as keyof Settings] as any)[setting];
            if (element instanceof HTMLInputElement) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                }
            } else if (element instanceof HTMLSelectElement) {
                element.value = value;
            }
        }
    }

    private setupUI() {
        // Create UI sections for each settings category
        const categories = ['nodes', 'edges', 'physics', 'rendering', 'bloom', 'animation', 'label', 'ar'] as const;
        
        const container = document.getElementById('settings-panel');
        if (!container) return;

        categories.forEach(category => {
            const section = document.createElement('div');
            section.className = 'settings-section';
            section.innerHTML = `<h3>${category.charAt(0).toUpperCase() + category.slice(1)} Settings</h3>`;

            const settings = this.settings[category];
            Object.entries(settings).forEach(([setting, value]) => {
                const settingElement = this.createSettingElement(category, setting, value);
                section.appendChild(settingElement);
            });

            container.appendChild(section);
        });
    }

    private createSettingElement(category: string, setting: string, value: any): HTMLElement {
        const wrapper = document.createElement('div');
        wrapper.className = 'setting-item';

        const label = document.createElement('label');
        label.textContent = this.formatSettingName(setting);

        const input = this.createInputElement(setting, value);
        input.id = `${category}-${setting}`;
        input.addEventListener('change', (e) => {
            const target = e.target as HTMLInputElement;
            const newValue = target.type === 'checkbox' ? target.checked : 
                            target.type === 'number' ? parseFloat(target.value) : 
                            target.value;
            this.updateSetting(category as keyof Settings, setting, newValue);
        });

        wrapper.appendChild(label);
        wrapper.appendChild(input);
        return wrapper;
    }

    private createInputElement(setting: string, value: any): HTMLInputElement | HTMLSelectElement {
        if (typeof value === 'boolean') {
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value;
            return input;
        } else if (typeof value === 'number') {
            const input = document.createElement('input');
            input.type = 'number';
            input.value = value.toString();
            input.step = '0.1';
            return input;
        } else if (typeof value === 'string') {
            if (setting === 'style' || setting === 'shape' || setting === 'texture') {
                const select = document.createElement('select');
                const options = this.getOptionsForSetting(setting);
                options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    option.selected = opt === value;
                    select.appendChild(option);
                });
                return select;
            } else {
                const input = document.createElement('input');
                input.type = setting.toLowerCase().includes('color') ? 'color' : 'text';
                input.value = value;
                return input;
            }
        }
        const input = document.createElement('input');
        input.type = 'text';
        input.value = value.toString();
        return input;
    }

    private getOptionsForSetting(setting: string): string[] {
        switch (setting) {
            case 'style':
                return ['solid', 'dashed', 'dotted', 'arrow'];
            case 'shape':
                return ['sphere', 'cube', 'cylinder', 'cone'];
            case 'texture':
                return ['none', 'smooth', 'rough', 'metallic'];
            default:
                return [];
        }
    }

    private formatSettingName(name: string): string {
        return name
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, str => str.toUpperCase());
    }

    dispose() {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
