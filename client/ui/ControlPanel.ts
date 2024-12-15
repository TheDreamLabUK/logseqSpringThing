/**
 * Control panel for visualization settings
 */

import { VisualizationSettings } from '../core/types';
import { settingsManager } from '../state/settings';
import { createLogger } from '../core/utils';

const logger = createLogger('ControlPanel');

export class ControlPanel {
    private container: HTMLDivElement;
    private settings: VisualizationSettings;
    private isExpanded = false;
    private feedbackTimeout: number | null = null;
    private statusIndicator: HTMLDivElement;
    
    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'control-panel';
        this.settings = { ...settingsManager.getSettings() };
        
        // Create status indicator
        this.statusIndicator = document.createElement('div');
        this.statusIndicator.className = 'connection-status';
        this.container.appendChild(this.statusIndicator);
        
        this.initializeUI();
        this.setupEventListeners();
        
        // Subscribe to settings updates
        settingsManager.addSettingsListener(this.onSettingsUpdate.bind(this));
        
        // Monitor WebSocket connection
        this.monitorConnection();
    }

    private showFeedback(message: string, type: 'success' | 'error' = 'success'): void {
        // Clear any existing feedback
        if (this.feedbackTimeout) {
            clearTimeout(this.feedbackTimeout);
            const existingFeedback = this.container.querySelector('.settings-feedback');
            if (existingFeedback) {
                existingFeedback.remove();
            }
        }

        // Create feedback element
        const feedback = document.createElement('div');
        feedback.className = `settings-feedback ${type}`;
        feedback.textContent = message;

        // Add to container before actions
        const actionsGroup = this.container.querySelector('.settings-actions');
        if (actionsGroup) {
            actionsGroup.insertAdjacentElement('beforebegin', feedback);
        }

        // Auto-remove after delay
        this.feedbackTimeout = window.setTimeout(() => {
            feedback.classList.add('fade-out');
            setTimeout(() => feedback.remove(), 300);
            this.feedbackTimeout = null;
        }, 3000);
    }

    private monitorConnection(): void {
        const updateStatus = (connected: boolean) => {
            this.statusIndicator.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            this.statusIndicator.title = connected ? 'Connected to server' : 'Disconnected from server';
        };

        // Initial status
        updateStatus(settingsManager.isConnected());

        // Listen for connection changes
        settingsManager.onConnectionChange((connected) => {
            updateStatus(connected);
            if (!connected) {
                this.showFeedback('Lost connection to server', 'error');
            }
        });
    }

    private onSettingsUpdate(newSettings: VisualizationSettings): void {
        this.settings = { ...newSettings };
        this.updateUIValues();
        logger.log('Settings updated from external source');
    }

    private initializeUI(): void {
        this.container.innerHTML = `
            <div class="control-panel-header">
                <h3>Graph Controls</h3>
                <button class="toggle-button">≡</button>
            </div>
            <div class="control-panel-content">
                <div class="settings-group">
                    <h4>Node Appearance</h4>
                    <div class="setting-item">
                        <label for="nodeSize">Node Size</label>
                        <input type="range" id="nodeSize" min="0.05" max="2" step="0.05" value="${this.settings.nodeSize}">
                        <span class="setting-value">${this.settings.nodeSize.toFixed(2)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="nodeColor">Color</label>
                        <input type="color" id="nodeColor" value="${this.settings.nodeColor}">
                    </div>
                    <div class="setting-item">
                        <label for="nodeOpacity">Opacity</label>
                        <input type="range" id="nodeOpacity" min="0" max="1" step="0.1" value="${this.settings.nodeOpacity}">
                        <span class="setting-value">${this.settings.nodeOpacity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="metalness">Metalness</label>
                        <input type="range" id="metalness" min="0" max="1" step="0.05" value="${this.settings.metalness}">
                        <span class="setting-value">${this.settings.metalness.toFixed(2)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="roughness">Roughness</label>
                        <input type="range" id="roughness" min="0" max="1" step="0.05" value="${this.settings.roughness}">
                        <span class="setting-value">${this.settings.roughness.toFixed(2)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="clearcoat">Clearcoat</label>
                        <input type="range" id="clearcoat" min="0" max="1" step="0.1" value="${this.settings.clearcoat}">
                        <span class="setting-value">${this.settings.clearcoat.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="enableInstancing">Enable Instancing</label>
                        <input type="checkbox" id="enableInstancing" ${this.settings.enableInstancing ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="materialType">Material Type</label>
                        <select id="materialType" value="${this.settings.materialType}">
                            <option value="physical">Physical</option>
                            <option value="basic">Basic</option>
                            <option value="phong">Phong</option>
                        </select>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Node Highlight</h4>
                    <div class="setting-item">
                        <label for="highlightColor">Highlight Color</label>
                        <input type="color" id="highlightColor" value="${this.settings.highlightColor}">
                    </div>
                    <div class="setting-item">
                        <label for="highlightDuration">Duration (ms)</label>
                        <input type="number" id="highlightDuration" min="0" max="2000" step="100" value="${this.settings.highlightDuration}">
                    </div>
                    <div class="setting-item">
                        <label for="enableHoverEffect">Enable Hover</label>
                        <input type="checkbox" id="enableHoverEffect" ${this.settings.enableHoverEffect ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="hoverScale">Hover Scale</label>
                        <input type="range" id="hoverScale" min="1" max="2" step="0.1" value="${this.settings.hoverScale}">
                        <span class="setting-value">${this.settings.hoverScale.toFixed(1)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Edge Appearance</h4>
                    <div class="setting-item">
                        <label for="edgeWidth">Edge Width</label>
                        <input type="range" id="edgeWidth" min="0.5" max="5" step="0.5" value="${this.settings.edgeWidth}">
                        <span class="setting-value">${this.settings.edgeWidth.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="edgeColor">Color</label>
                        <input type="color" id="edgeColor" value="${this.settings.edgeColor}">
                    </div>
                    <div class="setting-item">
                        <label for="edgeOpacity">Opacity</label>
                        <input type="range" id="edgeOpacity" min="0" max="1" step="0.1" value="${this.settings.edgeOpacity}">
                        <span class="setting-value">${this.settings.edgeOpacity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="enableArrows">Enable Arrows</label>
                        <input type="checkbox" id="enableArrows" ${this.settings.enableArrows ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="arrowSize">Arrow Size</label>
                        <input type="range" id="arrowSize" min="0.05" max="0.5" step="0.05" value="${this.settings.arrowSize}">
                        <span class="setting-value">${this.settings.arrowSize.toFixed(2)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Physics Settings</h4>
                    <div class="setting-item">
                        <label for="physicsEnabled">Enable Physics</label>
                        <input type="checkbox" id="physicsEnabled" ${this.settings.physicsEnabled ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="attractionStrength">Attraction</label>
                        <input type="range" id="attractionStrength" min="0" max="0.05" step="0.001" value="${this.settings.attractionStrength}">
                        <span class="setting-value">${this.settings.attractionStrength.toFixed(3)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="repulsionStrength">Repulsion</label>
                        <input type="range" id="repulsionStrength" min="0" max="3000" step="100" value="${this.settings.repulsionStrength}">
                        <span class="setting-value">${this.settings.repulsionStrength.toFixed(0)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="springStrength">Spring</label>
                        <input type="range" id="springStrength" min="0" max="0.05" step="0.001" value="${this.settings.springStrength}">
                        <span class="setting-value">${this.settings.springStrength.toFixed(3)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="damping">Damping</label>
                        <input type="range" id="damping" min="0" max="1" step="0.01" value="${this.settings.damping}">
                        <span class="setting-value">${this.settings.damping.toFixed(2)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="maxVelocity">Max Velocity</label>
                        <input type="range" id="maxVelocity" min="0.5" max="5" step="0.5" value="${this.settings.maxVelocity}">
                        <span class="setting-value">${this.settings.maxVelocity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="collisionRadius">Collision Radius</label>
                        <input type="range" id="collisionRadius" min="0.1" max="1" step="0.05" value="${this.settings.collisionRadius}">
                        <span class="setting-value">${this.settings.collisionRadius.toFixed(2)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Lighting</h4>
                    <div class="setting-item">
                        <label for="ambientLightIntensity">Ambient Light</label>
                        <input type="range" id="ambientLightIntensity" min="0" max="2" step="0.1" value="${this.settings.ambientLightIntensity}">
                        <span class="setting-value">${this.settings.ambientLightIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="directionalLightIntensity">Directional Light</label>
                        <input type="range" id="directionalLightIntensity" min="0" max="2" step="0.1" value="${this.settings.directionalLightIntensity}">
                        <span class="setting-value">${this.settings.directionalLightIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="environmentIntensity">Environment</label>
                        <input type="range" id="environmentIntensity" min="0" max="2" step="0.1" value="${this.settings.environmentIntensity}">
                        <span class="setting-value">${this.settings.environmentIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="backgroundColor">Background Color</label>
                        <input type="color" id="backgroundColor" value="${this.settings.backgroundColor}">
                    </div>
                    <div class="setting-item">
                        <label for="enableAmbientOcclusion">Ambient Occlusion</label>
                        <input type="checkbox" id="enableAmbientOcclusion" ${this.settings.enableAmbientOcclusion ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="enableShadows">Enable Shadows</label>
                        <input type="checkbox" id="enableShadows" ${this.settings.enableShadows ? 'checked' : ''}>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Visual Effects</h4>
                    <div class="setting-item">
                        <label for="enableBloom">Enable Bloom</label>
                        <input type="checkbox" id="enableBloom" ${this.settings.enableBloom ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="bloomIntensity">Bloom Intensity</label>
                        <input type="range" id="bloomIntensity" min="0" max="3" step="0.1" value="${this.settings.bloomIntensity}">
                        <span class="setting-value">${this.settings.bloomIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="bloomRadius">Bloom Radius</label>
                        <input type="range" id="bloomRadius" min="0" max="1" step="0.1" value="${this.settings.bloomRadius}">
                        <span class="setting-value">${this.settings.bloomRadius.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="nodeBloomStrength">Node Bloom</label>
                        <input type="range" id="nodeBloomStrength" min="0" max="1" step="0.1" value="${this.settings.nodeBloomStrength}">
                        <span class="setting-value">${this.settings.nodeBloomStrength.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="edgeBloomStrength">Edge Bloom</label>
                        <input type="range" id="edgeBloomStrength" min="0" max="1" step="0.1" value="${this.settings.edgeBloomStrength}">
                        <span class="setting-value">${this.settings.edgeBloomStrength.toFixed(1)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Motion & Animation</h4>
                    <div class="setting-item">
                        <label for="enableNodeAnimations">Node Animations</label>
                        <input type="checkbox" id="enableNodeAnimations" ${this.settings.enableNodeAnimations ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="enableMotionBlur">Motion Blur</label>
                        <input type="checkbox" id="enableMotionBlur" ${this.settings.enableMotionBlur ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="motionBlurStrength">Blur Strength</label>
                        <input type="range" id="motionBlurStrength" min="0" max="1" step="0.1" value="${this.settings.motionBlurStrength}">
                        <span class="setting-value">${this.settings.motionBlurStrength.toFixed(1)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Labels</h4>
                    <div class="setting-item">
                        <label for="showLabels">Show Labels</label>
                        <input type="checkbox" id="showLabels" ${this.settings.showLabels ? 'checked' : ''}>
                    </div>
                    <div class="setting-item">
                        <label for="labelSize">Label Size</label>
                        <input type="range" id="labelSize" min="0.5" max="2" step="0.1" value="${this.settings.labelSize}">
                        <span class="setting-value">${this.settings.labelSize.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="labelColor">Label Color</label>
                        <input type="color" id="labelColor" value="${this.settings.labelColor}">
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Performance</h4>
                    <div class="setting-item">
                        <label for="maxFps">Max FPS</label>
                        <input type="number" id="maxFps" min="30" max="144" step="1" value="${this.settings.maxFps}">
                    </div>
                    <div class="setting-item">
                        <label for="enableAntialiasing">Antialiasing</label>
                        <input type="checkbox" id="enableAntialiasing" ${this.settings.enableAntialiasing ? 'checked' : ''}>
                    </div>
                </div>

                <div class="settings-actions">
                    <button id="saveSettings" class="primary-button">Save Settings</button>
                    <button id="resetSettings" class="secondary-button">Reset to Defaults</button>
                </div>
            </div>
        `;

        document.body.appendChild(this.container);
    }

    private setupEventListeners(): void {
        // Toggle panel
        const toggleButton = this.container.querySelector('.toggle-button');
        toggleButton?.addEventListener('click', () => {
            this.isExpanded = !this.isExpanded;
            this.container.classList.toggle('expanded', this.isExpanded);
        });

        // Save button
        const saveButton = this.container.querySelector('#saveSettings');
        saveButton?.addEventListener('click', () => this.saveSettings());

        // Reset button
        const resetButton = this.container.querySelector('#resetSettings');
        resetButton?.addEventListener('click', () => this.resetSettings());

        // Node appearance settings
        this.setupRangeListener('nodeSize', 'nodeSize');
        this.setupColorListener('nodeColor', 'nodeColor');
        this.setupRangeListener('nodeOpacity', 'nodeOpacity');
        this.setupRangeListener('metalness', 'metalness');
        this.setupRangeListener('roughness', 'roughness');
        this.setupRangeListener('clearcoat', 'clearcoat');
        this.setupCheckboxListener('enableInstancing', 'enableInstancing');
        this.setupSelectListener('materialType', 'materialType');

        // Node highlight settings
        this.setupColorListener('highlightColor', 'highlightColor');
        this.setupNumberListener('highlightDuration', 'highlightDuration');
        this.setupCheckboxListener('enableHoverEffect', 'enableHoverEffect');
        this.setupRangeListener('hoverScale', 'hoverScale');

        // Edge settings
        this.setupRangeListener('edgeWidth', 'edgeWidth');
        this.setupColorListener('edgeColor', 'edgeColor');
        this.setupRangeListener('edgeOpacity', 'edgeOpacity');
        this.setupCheckboxListener('enableArrows', 'enableArrows');
        this.setupRangeListener('arrowSize', 'arrowSize');

        // Physics settings
        this.setupCheckboxListener('physicsEnabled', 'physicsEnabled');
        this.setupRangeListener('attractionStrength', 'attractionStrength');
        this.setupRangeListener('repulsionStrength', 'repulsionStrength');
        this.setupRangeListener('springStrength', 'springStrength');
        this.setupRangeListener('damping', 'damping');
        this.setupRangeListener('maxVelocity', 'maxVelocity');
        this.setupRangeListener('collisionRadius', 'collisionRadius');

        // Lighting settings
        this.setupRangeListener('ambientLightIntensity', 'ambientLightIntensity');
        this.setupRangeListener('directionalLightIntensity', 'directionalLightIntensity');
        this.setupRangeListener('environmentIntensity', 'environmentIntensity');
        this.setupColorListener('backgroundColor', 'backgroundColor');
        this.setupCheckboxListener('enableAmbientOcclusion', 'enableAmbientOcclusion');
        this.setupCheckboxListener('enableShadows', 'enableShadows');

        // Visual effects settings
        this.setupCheckboxListener('enableBloom', 'enableBloom');
        this.setupRangeListener('bloomIntensity', 'bloomIntensity');
        this.setupRangeListener('bloomRadius', 'bloomRadius');
        this.setupRangeListener('nodeBloomStrength', 'nodeBloomStrength');
        this.setupRangeListener('edgeBloomStrength', 'edgeBloomStrength');

        // Motion & animation settings
        this.setupCheckboxListener('enableNodeAnimations', 'enableNodeAnimations');
        this.setupCheckboxListener('enableMotionBlur', 'enableMotionBlur');
        this.setupRangeListener('motionBlurStrength', 'motionBlurStrength');

        // Label settings
        this.setupCheckboxListener('showLabels', 'showLabels');
        this.setupRangeListener('labelSize', 'labelSize');
        this.setupColorListener('labelColor', 'labelColor');

        // Performance settings
        this.setupNumberListener('maxFps', 'maxFps');
        this.setupCheckboxListener('enableAntialiasing', 'enableAntialiasing');
    }

    private setupRangeListener(elementId: string, settingKey: keyof VisualizationSettings): void {
        const element = this.container.querySelector(`#${elementId}`) as HTMLInputElement;
        if (element) {
            element.addEventListener('input', () => {
                (this.settings[settingKey] as number) = parseFloat(element.value);
                const valueDisplay = element.parentElement?.querySelector('.setting-value');
                if (valueDisplay) {
                    valueDisplay.textContent = parseFloat(element.value).toFixed(
                        element.step.includes('.') ? element.step.split('.')[1].length : 0
                    );
                }
                settingsManager.updateSettings(this.settings);
            });
        }
    }

    private setupColorListener(elementId: string, settingKey: keyof VisualizationSettings): void {
        const element = this.container.querySelector(`#${elementId}`) as HTMLInputElement;
        if (element) {
            element.addEventListener('input', () => {
                (this.settings[settingKey] as string) = element.value;
                settingsManager.updateSettings(this.settings);
            });
        }
    }

    private setupCheckboxListener(elementId: string, settingKey: keyof VisualizationSettings): void {
        const element = this.container.querySelector(`#${elementId}`) as HTMLInputElement;
        if (element) {
            element.addEventListener('change', () => {
                (this.settings[settingKey] as boolean) = element.checked;
                settingsManager.updateSettings(this.settings);
            });
        }
    }

    private setupNumberListener(elementId: string, settingKey: keyof VisualizationSettings): void {
        const element = this.container.querySelector(`#${elementId}`) as HTMLInputElement;
        if (element) {
            element.addEventListener('input', () => {
                (this.settings[settingKey] as number) = parseInt(element.value);
                settingsManager.updateSettings(this.settings);
            });
        }
    }

    private setupSelectListener(elementId: string, settingKey: keyof VisualizationSettings): void {
        const element = this.container.querySelector(`#${elementId}`) as HTMLSelectElement;
        if (element) {
            element.addEventListener('change', () => {
                (this.settings[settingKey] as string) = element.value;
                settingsManager.updateSettings(this.settings);
            });
        }
    }

    private updateUIValues(): void {
        // Update all input values to match current settings
        Object.entries(this.settings).forEach(([key, value]) => {
            const element = this.container.querySelector(`#${key}`) as HTMLInputElement | HTMLSelectElement;
            if (element) {
                if (element instanceof HTMLInputElement) {
                    if (element.type === 'checkbox') {
                        element.checked = value as boolean;
                    } else if (element.type === 'range' || element.type === 'number') {
                        element.value = value.toString();
                        const valueDisplay = element.parentElement?.querySelector('.setting-value');
                        if (valueDisplay) {
                            valueDisplay.textContent = typeof value === 'number' 
                                ? value.toFixed(element.step.includes('.') ? element.step.split('.')[1].length : 0)
                                : value.toString();
                        }
                    } else {
                        element.value = value as string;
                    }
                } else {
                    element.value = value as string;
                }
            }
        });
    }

    private async saveSettings(): Promise<void> {
        const saveButton = this.container.querySelector('#saveSettings') as HTMLButtonElement;
        if (saveButton) {
            saveButton.disabled = true;
            saveButton.textContent = 'Saving...';
        }

        try {
            await settingsManager.saveSettings();
            this.showFeedback('Settings saved successfully');
            logger.log('Settings saved successfully');
        } catch (error) {
            this.showFeedback('Failed to save settings', 'error');
            logger.error('Failed to save settings:', error);
        } finally {
            if (saveButton) {
                saveButton.disabled = false;
                saveButton.textContent = 'Save Settings';
            }
        }
    }

    private resetSettings(): void {
        const resetButton = this.container.querySelector('#resetSettings') as HTMLButtonElement;
        if (resetButton) {
            resetButton.disabled = true;
            resetButton.textContent = 'Resetting...';
        }

        try {
            settingsManager.resetToDefaults();
            this.showFeedback('Settings reset to defaults');
            logger.log('Settings reset to defaults');
        } catch (error) {
            this.showFeedback('Failed to reset settings', 'error');
            logger.error('Failed to reset settings:', error);
        } finally {
            if (resetButton) {
                resetButton.disabled = false;
                resetButton.textContent = 'Reset to Defaults';
            }
        }
    }

    // ... [Rest of the code remains the same] ...
}
