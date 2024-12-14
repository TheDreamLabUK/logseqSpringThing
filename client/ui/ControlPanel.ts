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
                        <input type="range" id="nodeSize" min="0.05" max="0.5" step="0.05" value="${this.settings.nodeSize}">
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
                        <label>
                            <input type="checkbox" id="enableArrows" ${this.settings.enableArrows ? 'checked' : ''}>
                            Show Arrows
                        </label>
                    </div>
                    <div class="setting-item arrow-setting ${this.settings.enableArrows ? '' : 'disabled'}">
                        <label for="arrowSize">Arrow Size</label>
                        <input type="range" id="arrowSize" min="0.1" max="1" step="0.05" value="${this.settings.arrowSize}">
                        <span class="setting-value">${this.settings.arrowSize.toFixed(2)}</span>
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
                        <label for="bloomIntensity">Bloom Intensity</label>
                        <input type="range" id="bloomIntensity" min="0" max="2" step="0.1" value="${this.settings.bloomIntensity}">
                        <span class="setting-value">${this.settings.bloomIntensity.toFixed(1)}</span>
                    </div>
                    <div class="setting-item bloom-setting ${this.settings.enableBloom ? '' : 'disabled'}">
                        <label for="bloomRadius">Bloom Radius</label>
                        <input type="range" id="bloomRadius" min="0" max="2" step="0.1" value="${this.settings.bloomRadius}">
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Animations</h4>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableNodeAnimations" ${this.settings.enableNodeAnimations ? 'checked' : ''}>
                            Node Animations
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="enableMotionBlur" ${this.settings.enableMotionBlur ? 'checked' : ''}>
                            Motion Blur
                        </label>
                    </div>
                    <div class="setting-item motion-setting ${this.settings.enableMotionBlur ? '' : 'disabled'}">
                        <label for="motionBlurStrength">Motion Blur Strength</label>
                        <input type="range" id="motionBlurStrength" min="0" max="1" step="0.1" value="${this.settings.motionBlurStrength}">
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
                        <label for="labelSize">Label Size</label>
                        <input type="range" id="labelSize" min="0.5" max="2" step="0.1" value="${this.settings.labelSize}">
                        <span class="setting-value">${this.settings.labelSize.toFixed(1)}</span>
                    </div>
                    <div class="setting-item label-setting ${this.settings.showLabels ? '' : 'disabled'}">
                        <label for="labelColor">Label Color</label>
                        <input type="color" id="labelColor" value="${this.settings.labelColor}">
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Performance</h4>
                    <div class="setting-item">
                        <label for="maxFps">Max FPS</label>
                        <input type="number" id="maxFps" min="30" max="144" value="${this.settings.maxFps}">
                    </div>
                </div>

                <div class="settings-group">
                    <h4>AR Settings</h4>
                    <div class="setting-item">
                        <label>Scene Understanding</label>
                        <div class="sub-settings">
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="enablePlaneDetection" ${this.settings.enablePlaneDetection ? 'checked' : ''}>
                                    Plane Detection
                                </label>
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="enableSceneUnderstanding" ${this.settings.enableSceneUnderstanding ? 'checked' : ''}>
                                    Scene Understanding
                                </label>
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="showPlaneOverlay" ${this.settings.showPlaneOverlay ? 'checked' : ''}>
                                    Show Plane Overlay
                                </label>
                            </div>
                            <div class="setting-item">
                                <label for="planeOpacity">Plane Opacity</label>
                                <input type="range" id="planeOpacity" min="0" max="1" step="0.1" value="${this.settings.planeOpacity}">
                            </div>
                            <div class="setting-item">
                                <label for="planeColor">Plane Color</label>
                                <input type="color" id="planeColor" value="${this.settings.planeColor}">
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="enableLightEstimation" ${this.settings.enableLightEstimation ? 'checked' : ''}>
                                    Light Estimation
                                </label>
                            </div>
                        </div>
                    </div>

                    <div class="setting-item">
                        <label>Hand Tracking</label>
                        <div class="sub-settings">
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="enableHandTracking" ${this.settings.enableHandTracking ? 'checked' : ''}>
                                    Enable Hand Tracking
                                </label>
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="handMeshEnabled" ${this.settings.handMeshEnabled ? 'checked' : ''}>
                                    Show Hand Mesh
                                </label>
                            </div>
                            <div class="setting-item">
                                <label for="handMeshColor">Hand Mesh Color</label>
                                <input type="color" id="handMeshColor" value="${this.settings.handMeshColor}">
                            </div>
                            <div class="setting-item">
                                <label for="handMeshOpacity">Hand Mesh Opacity</label>
                                <input type="range" id="handMeshOpacity" min="0" max="1" step="0.1" value="${this.settings.handMeshOpacity}">
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="handRayEnabled" ${this.settings.handRayEnabled ? 'checked' : ''}>
                                    Show Hand Ray
                                </label>
                            </div>
                            <div class="setting-item">
                                <label for="handRayColor">Hand Ray Color</label>
                                <input type="color" id="handRayColor" value="${this.settings.handRayColor}">
                            </div>
                        </div>
                    </div>

                    <div class="setting-item">
                        <label>Gesture Controls</label>
                        <div class="sub-settings">
                            <div class="setting-item">
                                <label for="gestureSmoothing">Gesture Smoothing</label>
                                <input type="range" id="gestureSmoothing" min="0" max="1" step="0.1" value="${this.settings.gestureSmoothing}">
                            </div>
                            <div class="setting-item">
                                <label for="pinchThreshold">Pinch Threshold</label>
                                <input type="range" id="pinchThreshold" min="0" max="0.05" step="0.001" value="${this.settings.pinchThreshold}">
                            </div>
                            <div class="setting-item">
                                <label for="dragThreshold">Drag Threshold</label>
                                <input type="range" id="dragThreshold" min="0" max="0.1" step="0.01" value="${this.settings.dragThreshold}">
                            </div>
                        </div>
                    </div>

                    <div class="setting-item">
                        <label>Haptics</label>
                        <div class="sub-settings">
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="enableHaptics" ${this.settings.enableHaptics ? 'checked' : ''}>
                                    Enable Haptics
                                </label>
                            </div>
                            <div class="setting-item">
                                <label for="hapticIntensity">Haptic Intensity</label>
                                <input type="range" id="hapticIntensity" min="0" max="1" step="0.1" value="${this.settings.hapticIntensity}">
                            </div>
                        </div>
                    </div>

                    <div class="setting-item">
                        <label>Room Scale</label>
                        <div class="sub-settings">
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="roomScale" ${this.settings.roomScale ? 'checked' : ''}>
                                    Room Scale Mode
                                </label>
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="snapToFloor" ${this.settings.snapToFloor ? 'checked' : ''}>
                                    Snap to Floor
                                </label>
                            </div>
                        </div>
                    </div>

                    <div class="setting-item">
                        <label>Passthrough</label>
                        <div class="sub-settings">
                            <div class="setting-item">
                                <label for="passthroughOpacity">Opacity</label>
                                <input type="range" id="passthroughOpacity" min="0" max="1" step="0.1" value="${this.settings.passthroughOpacity}">
                            </div>
                            <div class="setting-item">
                                <label for="passthroughBrightness">Brightness</label>
                                <input type="range" id="passthroughBrightness" min="0" max="2" step="0.1" value="${this.settings.passthroughBrightness}">
                            </div>
                            <div class="setting-item">
                                <label for="passthroughContrast">Contrast</label>
                                <input type="range" id="passthroughContrast" min="0" max="2" step="0.1" value="${this.settings.passthroughContrast}">
                            </div>
                            <div class="setting-item">
                                <label>
                                    <input type="checkbox" id="enablePassthroughPortal" ${this.settings.enablePassthroughPortal ? 'checked' : ''}>
                                    Enable Portal
                                </label>
                            </div>
                            <div class="setting-item portal-setting ${this.settings.enablePassthroughPortal ? '' : 'disabled'}">
                                <label for="portalSize">Portal Size</label>
                                <input type="range" id="portalSize" min="0.1" max="2" step="0.1" value="${this.settings.portalSize}">
                            </div>
                            <div class="setting-item portal-setting ${this.settings.enablePassthroughPortal ? '' : 'disabled'}">
                                <label for="portalEdgeColor">Portal Edge Color</label>
                                <input type="color" id="portalEdgeColor" value="${this.settings.portalEdgeColor}">
                            </div>
                            <div class="setting-item portal-setting ${this.settings.enablePassthroughPortal ? '' : 'disabled'}">
                                <label for="portalEdgeWidth">Portal Edge Width</label>
                                <input type="range" id="portalEdgeWidth" min="0.001" max="0.05" step="0.001" value="${this.settings.portalEdgeWidth}">
                            </div>
                        </div>
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

        // Edge settings
        this.setupInputListener('edgeWidth', 'number');
        this.setupInputListener('edgeColor', 'string');
        this.setupInputListener('edgeOpacity', 'number');
        
        const arrowsCheckbox = this.container.querySelector('#enableArrows') as HTMLInputElement;
        arrowsCheckbox?.addEventListener('change', () => {
            const arrowSettings = this.container.querySelectorAll('.arrow-setting');
            arrowSettings.forEach(setting => {
                setting.classList.toggle('disabled', !arrowsCheckbox.checked);
            });
            this.settings.enableArrows = arrowsCheckbox.checked;
        });
        this.setupInputListener('arrowSize', 'number');

        // Bloom settings
        const bloomCheckbox = this.container.querySelector('#enableBloom') as HTMLInputElement;
        bloomCheckbox?.addEventListener('change', () => {
            const bloomSettings = this.container.querySelectorAll('.bloom-setting');
            bloomSettings.forEach(setting => {
                setting.classList.toggle('disabled', !bloomCheckbox.checked);
            });
            this.settings.enableBloom = bloomCheckbox.checked;
        });
        this.setupInputListener('bloomIntensity', 'number');
        this.setupInputListener('bloomRadius', 'number');

        // Animation settings
        const nodeAnimCheckbox = this.container.querySelector('#enableNodeAnimations') as HTMLInputElement;
        nodeAnimCheckbox?.addEventListener('change', () => {
            this.settings.enableNodeAnimations = nodeAnimCheckbox.checked;
        });

        const motionBlurCheckbox = this.container.querySelector('#enableMotionBlur') as HTMLInputElement;
        motionBlurCheckbox?.addEventListener('change', () => {
            const motionSettings = this.container.querySelectorAll('.motion-setting');
            motionSettings.forEach(setting => {
                setting.classList.toggle('disabled', !motionBlurCheckbox.checked);
            });
            this.settings.enableMotionBlur = motionBlurCheckbox.checked;
        });
        this.setupInputListener('motionBlurStrength', 'number');

        // Label settings
        const labelCheckbox = this.container.querySelector('#showLabels') as HTMLInputElement;
        labelCheckbox?.addEventListener('change', () => {
            const labelSettings = this.container.querySelectorAll('.label-setting');
            labelSettings.forEach(setting => {
                setting.classList.toggle('disabled', !labelCheckbox.checked);
            });
            this.settings.showLabels = labelCheckbox.checked;
        });
        this.setupInputListener('labelSize', 'number');
        this.setupInputListener('labelColor', 'string');

        // Performance settings
        this.setupInputListener('maxFps', 'number');

        // AR settings
        const planeDetectionCheckbox = this.container.querySelector('#enablePlaneDetection') as HTMLInputElement;
        planeDetectionCheckbox?.addEventListener('change', () => {
            this.settings.enablePlaneDetection = planeDetectionCheckbox.checked;
        });

        const sceneUnderstandingCheckbox = this.container.querySelector('#enableSceneUnderstanding') as HTMLInputElement;
        sceneUnderstandingCheckbox?.addEventListener('change', () => {
            this.settings.enableSceneUnderstanding = sceneUnderstandingCheckbox.checked;
        });

        const showPlaneOverlayCheckbox = this.container.querySelector('#showPlaneOverlay') as HTMLInputElement;
        showPlaneOverlayCheckbox?.addEventListener('change', () => {
            this.settings.showPlaneOverlay = showPlaneOverlayCheckbox.checked;
        });

        this.setupInputListener('planeOpacity', 'number');
        this.setupInputListener('planeColor', 'string');

        const lightEstimationCheckbox = this.container.querySelector('#enableLightEstimation') as HTMLInputElement;
        lightEstimationCheckbox?.addEventListener('change', () => {
            this.settings.enableLightEstimation = lightEstimationCheckbox.checked;
        });

        const handTrackingCheckbox = this.container.querySelector('#enableHandTracking') as HTMLInputElement;
        handTrackingCheckbox?.addEventListener('change', () => {
            this.settings.enableHandTracking = handTrackingCheckbox.checked;
        });

        const handMeshEnabledCheckbox = this.container.querySelector('#handMeshEnabled') as HTMLInputElement;
        handMeshEnabledCheckbox?.addEventListener('change', () => {
            this.settings.handMeshEnabled = handMeshEnabledCheckbox.checked;
        });

        this.setupInputListener('handMeshColor', 'string');
        this.setupInputListener('handMeshOpacity', 'number');

        const handRayEnabledCheckbox = this.container.querySelector('#handRayEnabled') as HTMLInputElement;
        handRayEnabledCheckbox?.addEventListener('change', () => {
            this.settings.handRayEnabled = handRayEnabledCheckbox.checked;
        });

        this.setupInputListener('handRayColor', 'string');

        this.setupInputListener('gestureSmoothing', 'number');
        this.setupInputListener('pinchThreshold', 'number');
        this.setupInputListener('dragThreshold', 'number');

        const hapticsCheckbox = this.container.querySelector('#enableHaptics') as HTMLInputElement;
        hapticsCheckbox?.addEventListener('change', () => {
            this.settings.enableHaptics = hapticsCheckbox.checked;
        });

        this.setupInputListener('hapticIntensity', 'number');

        const roomScaleCheckbox = this.container.querySelector('#roomScale') as HTMLInputElement;
        roomScaleCheckbox?.addEventListener('change', () => {
            this.settings.roomScale = roomScaleCheckbox.checked;
        });

        const snapToFloorCheckbox = this.container.querySelector('#snapToFloor') as HTMLInputElement;
        snapToFloorCheckbox?.addEventListener('change', () => {
            this.settings.snapToFloor = snapToFloorCheckbox.checked;
        });

        this.setupInputListener('passthroughOpacity', 'number');
        this.setupInputListener('passthroughBrightness', 'number');
        this.setupInputListener('passthroughContrast', 'number');

        const passthroughPortalCheckbox = this.container.querySelector('#enablePassthroughPortal') as HTMLInputElement;
        passthroughPortalCheckbox?.addEventListener('change', () => {
            const portalSettings = this.container.querySelectorAll('.portal-setting');
            portalSettings.forEach(setting => {
                setting.classList.toggle('disabled', !passthroughPortalCheckbox.checked);
            });
            this.settings.enablePassthroughPortal = passthroughPortalCheckbox.checked;
        });

        this.setupInputListener('portalSize', 'number');
        this.setupInputListener('portalEdgeColor', 'string');
        this.setupInputListener('portalEdgeWidth', 'number');
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
        Object.entries(this.settings).forEach(([key, value]) => {
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
    }
}
