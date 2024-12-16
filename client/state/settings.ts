import { createLogger } from '../utils/logger';
import { Settings } from '../core/types';
import { convertObjectKeysToSnakeCase, convertObjectKeysToCamelCase } from '../core/utils';

const logger = createLogger('SettingsManager');

export class SettingsManager {
    private settings: Settings;
    private subscribers: Map<string, Map<string, Set<(value: any) => void>>> = new Map();

    constructor() {
        this.settings = this.getDefaultSettings();
        this.initializeSettings();
    }

    public getDefaultSettings(): Settings {
        return {
            nodes: {
                baseSize: 1.0,
                baseColor: '#4CAF50',
                opacity: 0.8,
                highlightColor: '#ff4444',
                clearcoat: 0.5,
                enableHoverEffect: true,
                enableInstancing: true,
                highlightDuration: 500,
                hoverScale: 1.2,
                materialType: 'phong',
                metalness: 0.5,
                roughness: 0.5,
                sizeByConnections: false,
                sizeRange: [0.5, 2.0]
            },
            edges: {
                baseWidth: 0.2,
                color: '#E0E0E0',
                opacity: 0.6,
                arrowSize: 0.2,
                enableArrows: true,
                widthRange: [0.1, 0.5]
            },
            rendering: {
                backgroundColor: '#212121',
                ambientLightIntensity: 0.5,
                directionalLightIntensity: 0.8,
                enableAmbientOcclusion: true,
                enableAntialiasing: true,
                enableShadows: true,
                environmentIntensity: 1.0
            },
            labels: {
                enableLabels: true,
                desktopFontSize: 14,
                textColor: '#FFFFFF'
            },
            clientDebug: {
                enabled: false,
                enableDataDebug: false,
                enableWebsocketDebug: false,
                logBinaryHeaders: false,
                logFullJson: false
            },
            animations: {
                enableMotionBlur: true,
                enableNodeAnimations: true,
                motionBlurStrength: 0.5,
                selectionWaveEnabled: true,
                pulseEnabled: true,
                rippleEnabled: true,
                edgeAnimationEnabled: true,
                flowParticlesEnabled: true
            },
            ar: {
                dragThreshold: 0.1,
                enableHandTracking: true,
                enableHaptics: true,
                enableLightEstimation: true,
                enablePassthroughPortal: false,
                enablePlaneDetection: true,
                enableSceneUnderstanding: true,
                gestureSsmoothing: 0.5,
                handMeshColor: '#ffffff',
                handMeshEnabled: true,
                handMeshOpacity: 0.5,
                handPointSize: 5,
                handRayColor: '#00ff00',
                handRayEnabled: true,
                handRayWidth: 2,
                hapticIntensity: 1.0,
                passthroughBrightness: 1.0,
                passthroughContrast: 1.0,
                passthroughOpacity: 0.8,
                pinchThreshold: 0.5,
                planeColor: '#808080',
                planeOpacity: 0.3,
                portalEdgeColor: '#00ff00',
                portalEdgeWidth: 0.02,
                portalSize: 2.0,
                roomScale: true,
                rotationThreshold: 0.1,
                showPlaneOverlay: true,
                snapToFloor: true
            },
            audio: {
                enableAmbientSounds: true,
                enableInteractionSounds: true,
                enableSpatialAudio: true
            },
            bloom: {
                edgeBloomStrength: 1.0,
                enabled: true,
                environmentBloomStrength: 0.5,
                nodeBloomStrength: 1.0,
                radius: 0.75,
                strength: 1.5
            },
            default: {
                apiClientTimeout: 30000,
                enableMetrics: true,
                enableRequestLogging: true,
                logFormat: 'json',
                logLevel: 'info',
                maxConcurrentRequests: 10,
                maxPayloadSize: 1048576,
                maxRetries: 3,
                metricsPort: 9090,
                retryDelay: 1000
            },
            github: {
                basePath: '',
                owner: '',
                rateLimit: true,
                repo: '',
                token: ''
            },
            network: {
                bindAddress: '0.0.0.0',
                domain: 'localhost',
                enableHttp2: true,
                enableRateLimiting: true,
                enableTls: false,
                maxRequestSize: 1048576,
                minTlsVersion: 'TLS1.2',
                port: 3000,
                rateLimitRequests: 100,
                rateLimitWindow: 60,
                tunnelId: ''
            },
            openai: {
                apiKey: '',
                baseUrl: 'https://api.openai.com/v1',
                model: 'gpt-4',
                rateLimit: 10,
                timeout: 30000
            },
            perplexity: {
                apiKey: '',
                apiUrl: 'https://api.perplexity.ai',
                frequencyPenalty: 0,
                maxTokens: 1000,
                model: 'mixtral-8x7b',
                prompt: '',
                rateLimit: 10,
                presencePenalty: 0,
                temperature: 0.7,
                timeout: 30000,
                topP: 1
            },
            physics: {
                attractionStrength: 1.0,
                boundsSize: 1000,
                collisionRadius: 5,
                damping: 0.5,
                enableBounds: true,
                enabled: true,
                iterations: 1,
                maxVelocity: 10,
                repulsionStrength: 100,
                springStrength: 0.1
            },
            ragflow: {
                apiKey: '',
                baseUrl: 'http://localhost:8000',
                maxRetries: 3,
                timeout: 30000
            },
            security: {
                allowedOrigins: ['*'],
                auditLogPath: './audit.log',
                cookieHttponly: true,
                cookieSamesite: 'Lax',
                cookieSecure: false,
                csrfTokenTimeout: 3600,
                enableAuditLogging: true,
                enableRequestValidation: true,
                sessionTimeout: 86400
            },
            serverDebug: {
                enableDataDebug: false,
                enableWebsocketDebug: false,
                enabled: false,
                logBinaryHeaders: false,
                logFullJson: false
            },
            websocket: {
                binaryChunkSize: 1000,
                compressionEnabled: true,
                compressionThreshold: 1024,
                heartbeatInterval: 30000,
                heartbeatTimeout: 60000,
                maxConnections: 1000,
                maxMessageSize: 1048576
            }
        };
    }

    private async initializeSettings(): Promise<void> {
        try {
            // Fetch settings by category instead of all at once
            const categories = ['nodes', 'edges', 'rendering', 'labels'] as const;
            
            for (const category of categories) {
                try {
                    logger.info(`Loading settings for category: ${category}`);
                    const response = await fetch(`/api/visualization/settings/${category}`);
                    if (!response.ok) {
                        throw new Error(`Failed to fetch ${category} settings: ${response.statusText}`);
                    }
                    const data = await response.json();
                    
                    // Update settings for this category
                    this.settings[category] = convertObjectKeysToCamelCase(data);
                    
                    // Notify subscribers for this category
                    Object.entries(this.settings[category]).forEach(([setting, value]) => {
                        this.notifySubscribers(category, setting, value);
                    });
                    
                    logger.info(`Successfully loaded settings for ${category}`);
                } catch (error) {
                    logger.error(`Error loading ${category} settings:`, error);
                    logger.info(`Using default settings for ${category}`);
                }
            }
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            // Fall back to default settings on error
            this.settings = this.getDefaultSettings();
        }
    }

    public async updateSetting(category: keyof Settings, setting: string, value: any): Promise<void> {
        try {
            // Update local settings first
            const categorySettings = this.settings[category] as Record<string, any>;
            const oldValue = categorySettings[setting];
            categorySettings[setting] = value;

            // Convert category and setting to snake_case for API
            const snakeCaseCategory = convertObjectKeysToSnakeCase({ [category]: null })[0];
            const snakeCaseSetting = convertObjectKeysToSnakeCase({ [setting]: null })[0];

            // Prepare the request body with snake_case
            const requestBody = {
                value: convertObjectKeysToSnakeCase(value)
            };

            // Send update to server
            const response = await fetch(
                `/api/visualization/${snakeCaseCategory}/${snakeCaseSetting}`,
                {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                }
            );

            if (!response.ok) {
                // Revert local change on error
                categorySettings[setting] = oldValue;
                throw new Error(`Failed to update setting: ${response.statusText}`);
            }

            // Notify subscribers of successful update
            this.notifySubscribers(category, setting, value);
        } catch (error) {
            logger.error('Error updating setting:', error);
            throw error;
        }
    }

    public async getSetting(category: keyof Settings, setting: string): Promise<any> {
        try {
            // Convert camelCase to snake_case for API request
            const snakeCaseCategory = convertObjectKeysToSnakeCase({ [category]: null })[0];
            const snakeCaseSetting = convertObjectKeysToSnakeCase({ [setting]: null })[0];

            const response = await fetch(`/api/visualization/${snakeCaseCategory}/${snakeCaseSetting}`);
            if (!response.ok) {
                throw new Error(`Failed to get setting: ${response.statusText}`);
            }

            const data = await response.json();
            const value = convertObjectKeysToCamelCase(data.value);

            // Update local setting
            const categorySettings = this.settings[category] as Record<string, any>;
            categorySettings[setting] = value;

            return value;
        } catch (error) {
            logger.error('Error getting setting:', error);
            throw error;
        }
    }

    private notifySubscribers<T>(category: string, setting: string, value: T): void {
        const categoryMap = this.subscribers.get(category);
        if (!categoryMap) return;

        const settingSet = categoryMap.get(setting);
        if (!settingSet) return;

        settingSet.forEach(listener => {
            try {
                listener(value);
            } catch (error) {
                logger.error(`Error in settings listener for ${category}.${setting}:`, error);
            }
        });
    }

    public subscribe<T>(category: string, setting: string, listener: (value: T) => void): () => void {
        if (!this.subscribers.has(category)) {
            this.subscribers.set(category, new Map());
        }
        const categoryMap = this.subscribers.get(category)!;
        
        if (!categoryMap.has(setting)) {
            categoryMap.set(setting, new Set());
        }
        const settingSet = categoryMap.get(setting)!;
        
        settingSet.add(listener);
        
        // Send current value immediately
        const currentValue = (this.settings[category as keyof Settings] as any)[setting];
        if (currentValue !== undefined) {
            try {
                listener(currentValue);
            } catch (error) {
                logger.error(`Error in initial settings listener for ${category}.${setting}:`, error);
            }
        }
        
        return () => {
            settingSet.delete(listener);
            if (settingSet.size === 0) {
                categoryMap.delete(setting);
            }
            if (categoryMap.size === 0) {
                this.subscribers.delete(category);
            }
        };
    }

    public getCurrentSettings(): Settings {
        return JSON.parse(JSON.stringify(this.settings));
    }

    public dispose(): void {
        this.subscribers.clear();
    }
}

// Create singleton instance
export const settingsManager = new SettingsManager();

// Re-export Settings interface
export type { Settings };
