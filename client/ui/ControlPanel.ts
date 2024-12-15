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
    
    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'control-panel';
        this.settings = { ...settingsManager.getSettings() };
        this.initializeUI();
        this.setupEventListeners();
        
        // Subscribe to settings updates
        settingsManager.addSettingsListener(this.onSettingsUpdate.bind(this));
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
                        <label for="highlightColor">Highlight Color</label>
                        <input type="color" id="highlightColor" value="${this.settings.highlightColor}">
                    </div>
                    <div class="setting-item">
                        <label for="highlightDuration">Highlight Duration (ms)</label>
                        <input type="number" id="highlightDuration" min="100" max="2000" step="100" value="${this.settings.highlightDuration}">
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableHoverEffect" ${this.settings.enableHoverEffect ? 'checked' : ''}>
                            Enable Hover Effect
                        </label>
                    </div>
                    <div class="setting-item hover-setting ${this.settings.enableHoverEffect ? '' : 'disabled'}">
                        <label for="hoverScale">Hover Scale</label>
                        <input type="range" id="hoverScale" min="1" max="2" step="0.1" value="${this.settings.hoverScale}">
                        <span class="setting-value">${this.settings.hoverScale.toFixed(1)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Edge Appearance</h4>
                    <div class="setting-item">
                        <label for="edgeWidth">Width</label>
                        <input type="range" id="edgeWidth" min="0.1" max="5" step="0.1" value="${this.settings.edgeWidth}">
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
                        <label>Edge Width Range</label>
                        <div class="range-inputs">
                            <input type="number" id="edgeWidthRangeMin" min="0.1" max="5" step="0.1" value="${this.settings.edgeWidthRange[0]}">
                            <span>to</span>
                            <input type="number" id="edgeWidthRangeMax" min="0.1" max="5" step="0.1" value="${this.settings.edgeWidthRange[1]}">
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Visual Effects</h4>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableBloom" ${this.settings.enableBloom ? 'checked' : ''}>
                            Enable Bloom
                        </label>
                    </div>
                    <div class="setting-item bloom-setting ${this.settings.enableBloom ? '' : 'disabled'}">
                        <label for="nodeBloomStrength">Node Bloom Strength</label>
                        <input type="range" id="nodeBloomStrength" min="0" max="1" step="0.1" value="${this.settings.nodeBloomStrength}">
                        <span class="setting-value">${this.settings.nodeBloomStrength.toFixed(1)}</span>
                    </div>
                    <div class="setting-item bloom-setting ${this.settings.enableBloom ? '' : 'disabled'}">
                        <label for="edgeBloomStrength">Edge Bloom Strength</label>
                        <input type="range" id="edgeBloomStrength" min="0" max="1" step="0.1" value="${this.settings.edgeBloomStrength}">
                        <span class="setting-value">${this.settings.edgeBloomStrength.toFixed(1)}</span>
                    </div>
                    <div class="setting-item bloom-setting ${this.settings.enableBloom ? '' : 'disabled'}">
                        <label for="environmentBloomStrength">Environment Bloom Strength</label>
                        <input type="range" id="environmentBloomStrength" min="0" max="1" step="0.1" value="${this.settings.environmentBloomStrength}">
                        <span class="setting-value">${this.settings.environmentBloomStrength.toFixed(1)}</span>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Labels</h4>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="showLabels" ${this.settings.showLabels ? 'checked' : ''}>
                            Show Labels
                        </label>
                    </div>
                    <div class="setting-item label-setting ${this.settings.showLabels ? '' : 'disabled'}">
                        <label for="labelColor">Label Color</label>
                        <input type="color" id="labelColor" value="${this.settings.labelColor}">
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Physics</h4>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="physicsEnabled" ${this.settings.physicsEnabled ? 'checked' : ''}>
                            Enable Physics
                        </label>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="attractionStrength">Attraction Strength</label>
                        <input type="range" id="attractionStrength" min="0" max="0.1" step="0.001" value="${this.settings.attractionStrength}">
                        <span class="setting-value">${this.settings.attractionStrength.toFixed(3)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="repulsionStrength">Repulsion Strength</label>
                        <input type="range" id="repulsionStrength" min="0" max="3000" step="100" value="${this.settings.repulsionStrength}">
                        <span class="setting-value">${this.settings.repulsionStrength.toFixed(0)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="springStrength">Spring Strength</label>
                        <input type="range" id="springStrength" min="0" max="0.1" step="0.001" value="${this.settings.springStrength}">
                        <span class="setting-value">${this.settings.springStrength.toFixed(3)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="damping">Damping</label>
                        <input type="range" id="damping" min="0" max="1" step="0.01" value="${this.settings.damping}">
                        <span class="setting-value">${this.settings.damping.toFixed(2)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="maxVelocity">Max Velocity</label>
                        <input type="range" id="maxVelocity" min="0" max="10" step="0.1" value="${this.settings.maxVelocity}">
                        <span class="setting-value">${this.settings.maxVelocity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="collisionRadius">Collision Radius</label>
                        <input type="range" id="collisionRadius" min="0" max="1" step="0.05" value="${this.settings.collisionRadius}">
                        <span class="setting-value">${this.settings.collisionRadius.toFixed(2)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="boundsSize">Bounds Size</label>
                        <input type="range" id="boundsSize" min="1" max="50" step="1" value="${this.settings.boundsSize}">
                        <span class="setting-value">${this.settings.boundsSize.toFixed(0)}</span>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label>
                            <input type="checkbox" id="enableBounds" ${this.settings.enableBounds ? 'checked' : ''}>
                            Enable Bounds
                        </label>
                    </div>
                    <div class="setting-item physics-setting ${this.settings.physicsEnabled ? '' : 'disabled'}">
                        <label for="iterations">Iterations</label>
                        <input type="number" id="iterations" min="100" max="1000" step="100" value="${this.settings.iterations}">
                    </div>
                </div>

                <div class="settings-group">
                    <h4>AR Settings</h4>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableHandTracking" ${this.settings.enableHandTracking ? 'checked' : ''}>
                            Enable Hand Tracking
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableHaptics" ${this.settings.enableHaptics ? 'checked' : ''}>
                            Enable Haptics
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enablePlaneDetection" ${this.settings.enablePlaneDetection ? 'checked' : ''}>
                            Enable Plane Detection
                        </label>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Rendering</h4>
                    <div class="setting-item">
                        <label for="ambientLightIntensity">Ambient Light Intensity</label>
                        <input type="range" id="ambientLightIntensity" min="0" max="2" step="0.1" value="${this.settings.ambientLightIntensity}">
                        <span class="setting-value">${this.settings.ambientLightIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="directionalLightIntensity">Directional Light Intensity</label>
                        <input type="range" id="directionalLightIntensity" min="0" max="2" step="0.1" value="${this.settings.directionalLightIntensity}">
                        <span class="setting-value">${this.settings.directionalLightIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label for="environmentIntensity">Environment Intensity</label>
                        <input type="range" id="environmentIntensity" min="0" max="2" step="0.1" value="${this.settings.environmentIntensity}">
                        <span class="setting-value">${this.settings.environmentIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableAmbientOcclusion" ${this.settings.enableAmbientOcclusion ? 'checked' : ''}>
                            Enable Ambient Occlusion
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableAntialiasing" ${this.settings.enableAntialiasing ? 'checked' : ''}>
                            Enable Antialiasing
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableShadows" ${this.settings.enableShadows ? 'checked' : ''}>
                            Enable Shadows
                        </label>
                    </div>
                    <div class="setting-item">
                        <label for="backgroundColor">Background Color</label>
                        <input type="color" id="backgroundColor" value="${this.settings.backgroundColor}">
                    </div>
                </div>

                <button class="save-button">Save Changes</button>
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
        const saveButton = this.container.querySelector('.save-button');
        saveButton?.addEventListener('click', this.saveSettings.bind(this));

        // Node settings
        this.setupInputListener('nodeSize', 'number');
        this.setupInputListener('nodeColor', 'string');
        this.setupInputListener('nodeOpacity', 'number');
        this.setupInputListener('metalness', 'number');
        this.setupInputListener('roughness', 'number');
        this.setupInputListener('clearcoat', 'number');
        this.setupInputListener('highlightColor', 'string');
        this.setupInputListener('highlightDuration', 'number');
        
        const hoverEffectCheckbox = this.container.querySelector('#enableHoverEffect') as HTMLInputElement;
        hoverEffectCheckbox?.addEventListener('change', () => {
            const hoverSettings = this.container.querySelectorAll('.hover-setting');
            hoverSettings.forEach(setting => {
                setting.classList.toggle('disabled', !hoverEffectCheckbox.checked);
            });
            this.settings.enableHoverEffect = hoverEffectCheckbox.checked;
        });
        this.setupInputListener('hoverScale', 'number');

        // Edge settings
        this.setupInputListener('edgeWidth', 'number');
        this.setupInputListener('edgeColor', 'string');
        this.setupInputListener('edgeOpacity', 'number');
        
        // Edge width range
        const edgeWidthRangeMin = this.container.querySelector('#edgeWidthRangeMin') as HTMLInputElement;
        const edgeWidthRangeMax = this.container.querySelector('#edgeWidthRangeMax') as HTMLInputElement;
        
        edgeWidthRangeMin?.addEventListener('input', () => {
            this.settings.edgeWidthRange = [parseFloat(edgeWidthRangeMin.value), this.settings.edgeWidthRange[1]];
        });
        
        edgeWidthRangeMax?.addEventListener('input', () => {
            this.settings.edgeWidthRange = [this.settings.edgeWidthRange[0], parseFloat(edgeWidthRangeMax.value)];
        });

        // Bloom settings
        const bloomCheckbox = this.container.querySelector('#enableBloom') as HTMLInputElement;
        bloomCheckbox?.addEventListener('change', () => {
            const bloomSettings = this.container.querySelectorAll('.bloom-setting');
            bloomSettings.forEach(setting => {
                setting.classList.toggle('disabled', !bloomCheckbox.checked);
            });
            this.settings.enableBloom = bloomCheckbox.checked;
        });
        this.setupInputListener('nodeBloomStrength', 'number');
        this.setupInputListener('edgeBloomStrength', 'number');
        this.setupInputListener('environmentBloomStrength', 'number');

        // Label settings
        const labelCheckbox = this.container.querySelector('#showLabels') as HTMLInputElement;
        labelCheckbox?.addEventListener('change', () => {
            const labelSettings = this.container.querySelectorAll('.label-setting');
            labelSettings.forEach(setting => {
                setting.classList.toggle('disabled', !labelCheckbox.checked);
            });
            this.settings.showLabels = labelCheckbox.checked;
        });
        this.setupInputListener('labelColor', 'string');

        // Physics settings
        const physicsCheckbox = this.container.querySelector('#physicsEnabled') as HTMLInputElement;
        physicsCheckbox?.addEventListener('change', () => {
            const physicsSettings = this.container.querySelectorAll('.physics-setting');
            physicsSettings.forEach(setting => {
                setting.classList.toggle('disabled', !physicsCheckbox.checked);
            });
            this.settings.physicsEnabled = physicsCheckbox.checked;
        });
        
        this.setupInputListener('attractionStrength', 'number');
        this.setupInputListener('repulsionStrength', 'number');
        this.setupInputListener('springStrength', 'number');
        this.setupInputListener('damping', 'number');
        this.setupInputListener('maxVelocity', 'number');
        this.setupInputListener('collisionRadius', 'number');
        this.setupInputListener('boundsSize', 'number');
        
        const boundsCheckbox = this.container.querySelector('#enableBounds') as HTMLInputElement;
        boundsCheckbox?.addEventListener('change', () => {
            this.settings.enableBounds = boundsCheckbox.checked;
        });
        
        this.setupInputListener('iterations', 'number');

        // AR settings
        const handTrackingCheckbox = this.container.querySelector('#enableHandTracking') as HTMLInputElement;
        handTrackingCheckbox?.addEventListener('change', () => {
            this.settings.enableHandTracking = handTrackingCheckbox.checked;
        });

        const hapticsCheckbox = this.container.querySelector('#enableHaptics') as HTMLInputElement;
        hapticsCheckbox?.addEventListener('change', () => {
            this.settings.enableHaptics = hapticsCheckbox.checked;
        });

        const planeDetectionCheckbox = this.container.querySelector('#enablePlaneDetection') as HTMLInputElement;
        planeDetectionCheckbox?.addEventListener('change', () => {
            this.settings.enablePlaneDetection = planeDetectionCheckbox.checked;
        });

        // Rendering settings
        this.setupInputListener('ambientLightIntensity', 'number');
        this.setupInputListener('directionalLightIntensity', 'number');
        this.setupInputListener('environmentIntensity', 'number');
        
        const ambientOcclusionCheckbox = this.container.querySelector('#enableAmbientOcclusion') as HTMLInputElement;
        ambientOcclusionCheckbox?.addEventListener('change', () => {
            this.settings.enableAmbientOcclusion = ambientOcclusionCheckbox.checked;
        });

        const antialiasingCheckbox = this.container.querySelector('#enableAntialiasing') as HTMLInputElement;
        antialiasingCheckbox?.addEventListener('change', () => {
            this.settings.enableAntialiasing = antialiasingCheckbox.checked;
        });

        const shadowsCheckbox = this.container.querySelector('#enableShadows') as HTMLInputElement;
        shadowsCheckbox?.addEventListener('change', () => {
            this.settings.enableShadows = shadowsCheckbox.checked;
        });

        this.setupInputListener('backgroundColor', 'string');
    }

    private setupInputListener(id: string, type: 'number' | 'string'): void {
        const input = this.container.querySelector(`#${id}`) as HTMLInputElement;
        const valueDisplay = input?.parentElement?.querySelector('.setting-value');
        
        input?.addEventListener('input', () => {
            const value = type === 'number' ? parseFloat(input.value) : input.value;
            (this.settings as any)[id] = value;
            
            // Update value display
            if (valueDisplay && type === 'number') {
                const decimals = input.step.includes('.') ? input.step.split('.')[1].length : 0;
                valueDisplay.textContent = typeof value === 'number' ? value.toFixed(decimals) : Number(value).toFixed(decimals);
            }
        });
    }

    private async saveSettings(): Promise<void> {
        try {
            await settingsManager.updateSettings(this.settings);
            logger.log('Settings saved successfully');
        } catch (error) {
            logger.error('Failed to save settings:', error);
        }
    }

    private onSettingsUpdate(newSettings: VisualizationSettings): void {
        this.settings = { ...newSettings };
        this.updateUIValues();
    }

    private updateUIValues(): void {
        // Update all input values
        Object.entries(this.settings).forEach(([key, value]) => {
            if (typeof value === 'object') return; // Skip nested objects (like github, openai, etc.)
            
            const input = this.container.querySelector(`#${key}`) as HTMLInputElement;
            const valueDisplay = input?.parentElement?.querySelector('.setting-value');
            
            if (input) {
                if (input.type === 'checkbox') {
                    input.checked = value as boolean;
                } else {
                    input.value = value.toString();
                    
                    // Update value display
                    if (valueDisplay && typeof value === 'number') {
                        const decimals = input.step.includes('.') ? input.step.split('.')[1].length : 0;
                        valueDisplay.textContent = value.toFixed(decimals);
                    }
                }
            }
        });

        // Update edge width range inputs
        const edgeWidthRangeMin = this.container.querySelector('#edgeWidthRangeMin') as HTMLInputElement;
        const edgeWidthRangeMax = this.container.querySelector('#edgeWidthRangeMax') as HTMLInputElement;
        if (edgeWidthRangeMin && edgeWidthRangeMax && Array.isArray(this.settings.edgeWidthRange)) {
            edgeWidthRangeMin.value = this.settings.edgeWidthRange[0].toString();
            edgeWidthRangeMax.value = this.settings.edgeWidthRange[1].toString();
        }

        // Update disabled states
        const updateDisabledState = (checkboxId: string, settingClass: string) => {
            const checkbox = this.container.querySelector(`#${checkboxId}`) as HTMLInputElement;
            const settings = this.container.querySelectorAll(`.${settingClass}`);
            if (checkbox && settings) {
                settings.forEach(setting => {
                    setting.classList.toggle('disabled', !checkbox.checked);
                });
            }
        };

        updateDisabledState('enableHoverEffect', 'hover-setting');
        updateDisabledState('enableBloom', 'bloom-setting');
        updateDisabledState('showLabels', 'label-setting');
        updateDisabledState('physicsEnabled', 'physics-setting');
    }
}
