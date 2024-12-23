import { Settings } from '../types/settings';
import { settingsManager } from '../state/settings';
import { platformManager } from '../platform/platformManager';
import { createLogger } from '../core/logger';
import {
    SettingsCategory,
    SettingsPath,
    SettingValue,
    formatSettingName,
    getSettingInputType,
    getStepValue,
    isSettingsObject,
    getAllSettingPaths
} from '../types/settings/utils';
import './ControlPanel.css';
import { NodeManager } from '../rendering/nodes';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLElement;
    private settings: Settings;
    private unsubscribers: Array<() => void> = [];

    constructor(container: HTMLElement) {
        this.container = container;
        this.settings = settingsManager.getCurrentSettings();
        this.addRandomizeButton();
        this.initializePanel();
        this.setupSettingsSubscriptions();
    }

    private addRandomizeButton(): void {
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'randomize-button-container';

        const button = document.createElement('button');
        button.textContent = 'Randomize Node Positions';
        button.className = 'randomize-button';
        button.onclick = () => {
            const nodeManager = NodeManager.getInstance();
            const nodes = nodeManager.getCurrentNodes();
            
            const radius = 50;
            nodes.forEach(node => {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = radius * Math.cbrt(Math.random());

                node.data.position = {
                    x: r * Math.sin(phi) * Math.cos(theta),
                    y: r * Math.sin(phi) * Math.sin(theta),
                    z: r * Math.cos(phi)
                };
            });

            nodeManager.updateNodes(nodes);
            logger.info('Node positions randomized');
        };

        buttonContainer.appendChild(button);
        this.container.insertBefore(buttonContainer, this.container.firstChild);
    }

    private initializePanel(): void {
        // Create main sections
        this.createSettingsGroup('visualization', 'Visualization');
        
        if (platformManager.getCapabilities().xrSupported) {
            this.createXRSettingsSection();
        }
        
        this.createSettingsGroup('system', 'System');
    }

    private createXRSettingsSection(): void {
        const xrSection = document.createElement('div');
        xrSection.className = 'settings-group';
        
        const title = document.createElement('h3');
        title.textContent = 'XR Settings';
        xrSection.appendChild(title);

        // Add XR mode toggle
        const xrToggle = document.createElement('button');
        xrToggle.id = 'xr-toggle';
        xrToggle.textContent = 'Enter XR';
        xrToggle.className = 'xr-toggle-button';
        xrToggle.addEventListener('click', () => {
            window.dispatchEvent(new CustomEvent('toggleXR'));
        });
        xrSection.appendChild(xrToggle);

        // Create subsections for XR settings
        const subsections = [
            { key: 'input', title: 'Input & Interaction' },
            { key: 'visuals', title: 'Visual Settings' },
            { key: 'environment', title: 'Environment' },
            { key: 'passthrough', title: 'Passthrough' }
        ];

        subsections.forEach(({ key, title }) => {
            const subsection = document.createElement('div');
            subsection.className = 'settings-subsection';
            
            const subtitle = document.createElement('h4');
            subtitle.textContent = title;
            subsection.appendChild(subtitle);

            const settings = this.settings.xr[key as keyof typeof this.settings.xr];
            if (settings && typeof settings === 'object') {
                this.createSettingsElements(subsection, `xr.${key}`, settings);
            }
            
            xrSection.appendChild(subsection);
        });

        this.container.appendChild(xrSection);
    }

    private createSettingsGroup(category: SettingsCategory, title: string): void {
        const group = document.createElement('div');
        group.className = 'settings-group';
        
        const groupTitle = document.createElement('h3');
        groupTitle.textContent = title;
        group.appendChild(groupTitle);

        const categorySettings = this.settings[category];
        Object.entries(categorySettings).forEach(([key, value]) => {
            if (isSettingsObject(value)) {
                const subsection = document.createElement('div');
                subsection.className = 'settings-subsection';
                
                const subtitle = document.createElement('h4');
                subtitle.textContent = formatSettingName(key);
                subsection.appendChild(subtitle);

                this.createSettingsElements(subsection, `${category}.${key}`, value);
                group.appendChild(subsection);
            }
        });

        this.container.appendChild(group);
    }

    private createSettingsElements(container: HTMLElement, path: SettingsPath, settings: Record<string, any>): void {
        Object.entries(settings).forEach(([key, value]) => {
            if (!isSettingsObject(value)) {
                const settingElement = this.createSettingElement(`${path}.${key}` as SettingsPath, value);
                container.appendChild(settingElement);
            }
        });
    }

    private createSettingElement(path: SettingsPath, value: SettingValue): HTMLElement {
        const container = document.createElement('div');
        container.className = 'setting-item';

        const label = document.createElement('label');
        label.textContent = formatSettingName(path.split('.').pop()!);
        container.appendChild(label);

        const input = this.createInputElement(path, value);
        container.appendChild(input);

        return container;
    }

    private createInputElement(path: SettingsPath, value: SettingValue): HTMLElement {
        const input = document.createElement('input');
        const inputType = getSettingInputType(value);
        input.type = inputType;

        switch (inputType) {
            case 'checkbox':
                input.checked = value as boolean;
                break;
            case 'number':
                input.value = String(value);
                input.step = getStepValue(path.split('.').pop()!);
                break;
            default:
                input.value = String(value);
        }

        input.id = path;
        input.addEventListener('change', (event) => this.handleSettingChange(path, event));
        return input;
    }

    private async handleSettingChange(path: SettingsPath, event: Event): Promise<void> {
        const target = event.target as HTMLInputElement;
        let value: SettingValue;

        switch (target.type) {
            case 'checkbox':
                value = target.checked;
                break;
            case 'number':
                value = parseFloat(target.value);
                break;
            default:
                value = target.value;
        }

        try {
            await settingsManager.updateSetting(path, value);
            logger.info(`Updated setting ${path} to ${value}`);
        } catch (error) {
            logger.error(`Failed to update setting ${path}:`, error);
            this.revertSettingValue(path, target);
        }
    }

    private revertSettingValue(path: SettingsPath, input: HTMLInputElement): void {
        const value = settingsManager.get(path);
        if (input.type === 'checkbox') {
            input.checked = value as boolean;
        } else {
            input.value = String(value);
        }
    }

    private setupSettingsSubscriptions(): void {
        const paths = getAllSettingPaths(this.settings);
        paths.forEach(path => {
            const unsubscribe = settingsManager.subscribe(path, (value) => {
                this.updateSettingElement(path, value);
            });
            this.unsubscribers.push(unsubscribe);
        });
    }

    private updateSettingElement(path: SettingsPath, value: SettingValue): void {
        const input = document.getElementById(path) as HTMLInputElement;
        if (input) {
            if (input.type === 'checkbox') {
                input.checked = value as boolean;
            } else {
                input.value = String(value);
            }
        }
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.unsubscribers = [];
    }
}
