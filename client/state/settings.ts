import { createLogger } from '../utils/logger';
import { Settings, SettingCategory, UpdateSettingsMessage } from '../core/types';
import { WebSocketService } from '../websocket/websocketService';
import { convertObjectKeysToSnakeCase, convertObjectKeysToCamelCase } from '../core/utils';

const logger = createLogger('SettingsManager');

export class SettingsManager {
    private settings: Settings;
    private subscribers: Map<string, Map<string, Set<(value: any) => void>>> = new Map();
    private connectionSubscribers: Set<(connected: boolean) => void> = new Set();
    private connected: boolean = false;
    private webSocketService: WebSocketService;

    constructor(webSocketService: WebSocketService) {
        this.settings = this.getDefaultSettings();
        this.webSocketService = webSocketService;
        this.setupWebSocketHandlers();
        this.initializeSettings();
    }

private getDefaultSettings(): Settings {
    return {
        animations: {
            enableMotionBlur: false,
            enableNodeAnimations: false,
            motionBlurStrength: 0.4,
            selectionWaveEnabled: false,
            pulseEnabled: false,
            rippleEnabled: false,
            edgeAnimationEnabled: false,
            flowParticlesEnabled: false
        },
        ar: {
            dragThreshold: 0.04,
            enableHandTracking: true,
            enableHaptics: true,
            enableLightEstimation: true,
            enablePassthroughPortal: false,
            enablePlaneDetection: true,
            enableSceneUnderstanding: true,
            gestureSsmoothing: 0.9,
            handMeshColor: '#FFD700',
            handMeshEnabled: true,
            handMeshOpacity: 0.3,
            handPointSize: 0.01,
            handRayColor: '#FFD700',
            handRayEnabled: true,
            handRayWidth: 0.002,
            hapticIntensity: 0.7,
            passthroughBrightness: 1,
            passthroughContrast: 1,
            passthroughOpacity: 1,
            pinchThreshold: 0.015,
            planeColor: '#4A90E2',
            planeOpacity: 0.3,
            portalEdgeColor: '#FFD700',
            portalEdgeWidth: 0.02,
            portalSize: 1,
            roomScale: true,
            rotationThreshold: 0.08,
            showPlaneOverlay: true,
            snapToFloor: true
        },
        audio: {
            enableAmbientSounds: false,
            enableInteractionSounds: false,
            enableSpatialAudio: false
        },
        bloom: {
            edgeBloomStrength: 0.3,
            enabled: false,
            environmentBloomStrength: 0.5,
            nodeBloomStrength: 0.2,
            radius: 0.5,
            strength: 1.8
        },
        clientDebug: {
            enableDataDebug: false,
            enableWebsocketDebug: false,
            enabled: false,
            logBinaryHeaders: false,
            logFullJson: false
        },
        default: {
            apiClientTimeout: 30,
            enableMetrics: true,
            enableRequestLogging: true,
            logFormat: 'json',
            logLevel: 'debug',
            maxConcurrentRequests: 5,
            maxPayloadSize: 5242880,
            maxRetries: 3,
            metricsPort: 9090,
            retryDelay: 5
        },
        edges: {
            arrowSize: 0.15,
            baseWidth: 2,
            color: '#917f18',
            enableArrows: false,
            opacity: 0.6,
            widthRange: [1, 3]
        },
        github: {
            basePath: 'default_path',
            owner: 'default_owner',
            rateLimit: true,
            repo: 'default_repo',
            token: 'default_token'
        },
        labels: {
            desktopFontSize: 48,
            enableLabels: true,
            textColor: '#FFFFFF'
        },
        network: {
            bindAddress: '0.0.0.0',
            domain: 'localhost',
            enableHttp2: false,
            enableRateLimiting: true,
            enableTls: false,
            maxRequestSize: 10485760,
            minTlsVersion: '',
            port: 3000,
            rateLimitRequests: 100,
            rateLimitWindow: 60,
            tunnelId: 'dummy'
        },
        nodes: {
            baseColor: '#c3ab6f',
            baseSize: 1,
            clearcoat: 1,
            enableHoverEffect: false,
            enableInstancing: false,
            highlightColor: '#822626',
            highlightDuration: 300,
            hoverScale: 1.2,
            materialType: 'basic',
            metalness: 0.3,
            opacity: 0.4,
            roughness: 0.35,
            sizeByConnections: true,
            sizeRange: [1, 5]
        },
        openai: {
            apiKey: 'default_openai_key',
            baseUrl: 'wss://api.openai.com/v1/realtime',
            model: 'gpt-4o-realtime-preview-2024-10-01',
            rateLimit: 100,
            timeout: 30
        },
        perplexity: {
            apiKey: 'default_perplexity_key',
            apiUrl: 'https://api.perplexity.ai/chat/completions',
            frequencyPenalty: 1.0,
            maxTokens: 4096,
            model: 'llama-3.1-sonar-small-128k-online',
            prompt: '',
            rateLimit: 100,
            presencePenalty: 0.0,
            temperature: 0.5,
            timeout: 30,
            topP: 0.9
        },
        physics: {
            attractionStrength: 0.015,
            boundsSize: 12,
            collisionRadius: 0.25,
            damping: 0.88,
            enableBounds: true,
            enabled: false,
            iterations: 500,
            maxVelocity: 2.5,
            repulsionStrength: 1500,
            springStrength: 0.018
        },
        ragflow: {
            apiKey: 'default_ragflow_key',
            baseUrl: 'http://ragflow-server/v1/',
            maxRetries: 3,
            timeout: 30
        },
        rendering: {
            ambientLightIntensity: 0.7,
            backgroundColor: '#000000',
            directionalLightIntensity: 1,
            enableAmbientOcclusion: false,
            enableAntialiasing: true,
            enableShadows: false,
            environmentIntensity: 1.2
        },
        security: {
            allowedOrigins: [],
            auditLogPath: '/app/logs/audit.log',
            cookieHttponly: true,
            cookieSamesite: 'Strict',
            cookieSecure: true,
            csrfTokenTimeout: 3600,
            enableAuditLogging: true,
            enableRequestValidation: true,
            sessionTimeout: 3600
        },
        serverDebug: {
            enableDataDebug: false,
            enableWebsocketDebug: false,
            enabled: true,
            logBinaryHeaders: false,
            logFullJson: false
        },
        websocket: {
            binaryChunkSize: 65536,
            compressionEnabled: true,
            compressionThreshold: 1024,
            heartbeatInterval: 15000,
            heartbeatTimeout: 60000,
            maxConnections: 1000,
            maxMessageSize: 100485760
        }
    };
}

    private setupWebSocketHandlers(): void {
        this.webSocketService.on('settingsUpdated', (data: any) => {
            if (data && data.settings) {
                // Convert received snake_case settings to camelCase
                const camelCaseSettings = convertObjectKeysToCamelCase(data.settings);
                this.updateSettingsFromServer(camelCaseSettings);
            }
        });

        this.webSocketService.onConnectionChange((connected: boolean) => {
            this.setConnectionStatus(connected);
        });
    }

    private async initializeSettings(): Promise<void> {
        try {
            await this.loadAllSettings();
            this.setConnectionStatus(true);
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            this.setConnectionStatus(false);
        }
    }

    private updateSettingsFromServer(newSettings: Settings): void {
        Object.entries(newSettings).forEach(([category, categorySettings]) => {
            Object.entries(categorySettings).forEach(([setting, value]) => {
                (this.settings[category as keyof Settings] as any)[setting] = value;
                this.notifySubscribers(category, setting, value);
            });
        });
    }

    private async getSetting(category: keyof Settings, setting: string): Promise<any> {
        try {
            // Convert camelCase to snake_case for API request
            const snakeCaseCategory = category.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
            const snakeCaseSetting = setting.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
            
            const response = await fetch(`/api/visualization/${snakeCaseCategory}/${snakeCaseSetting}`);
            if (!response.ok) {
                throw new Error(`Failed to get setting: ${response.statusText}`);
            }
            const data = await response.json();
            return data.value;
        } catch (error) {
            logger.error(`Failed to get ${String(category)}.${setting}:`, error);
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

    public async updateSetting(category: keyof Settings, setting: string, value: any): Promise<void> {
        try {
            // Convert camelCase to snake_case for API request
            const snakeCaseCategory = category.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
            const snakeCaseSetting = setting.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
            
            const response = await fetch(`/api/visualization/${snakeCaseCategory}/${snakeCaseSetting}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ value })
            });

            if (!response.ok) {
                throw new Error(`Failed to update setting: ${response.statusText}`);
            }

            const result = await response.json();
            const newValue = result.value;

            (this.settings[category] as any)[setting] = newValue;
            this.notifySubscribers(category, setting, newValue);

            if (this.webSocketService.isConnectedToServer()) {
                const message: UpdateSettingsMessage = {
                    type: 'updateSettings',
                    data: { 
                        settings: convertObjectKeysToSnakeCase(this.settings)
                    }
                };
                this.webSocketService.send(message);
            }
            
            logger.info(`Updated ${String(category)}.${setting} to:`, newValue);
        } catch (error) {
            logger.error(`Failed to update ${String(category)}.${setting}:`, error);
            throw error;
        }
    }

    public async loadAllSettings(): Promise<void> {
        const categories = Object.keys(this.settings) as Array<keyof Settings>;
        
        for (const category of categories) {
            const settings = Object.keys(this.settings[category]);
            for (const setting of settings) {
                try {
                    const value = await this.getSetting(category, setting);
                    (this.settings[category] as any)[setting] = value;
                    this.notifySubscribers(category, setting, value);
                } catch (error) {
                    logger.error(`Failed to load ${String(category)}.${setting}:`, error);
                }
            }
        }

        if (this.webSocketService.isConnectedToServer()) {
            const message: UpdateSettingsMessage = {
                type: 'updateSettings',
                data: { 
                    settings: convertObjectKeysToSnakeCase(this.settings)
                }
            };
            this.webSocketService.send(message);
        }
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
        
        const currentValue = (this.settings[category as keyof Settings] as any)[setting];
        if (currentValue !== undefined) {
            listener(currentValue);
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

    public subscribeToConnection(listener: (connected: boolean) => void): () => void {
        this.connectionSubscribers.add(listener);
        listener(this.connected);
        return () => {
            this.connectionSubscribers.delete(listener);
        };
    }

    public setConnectionStatus(connected: boolean): void {
        this.connected = connected;
        this.connectionSubscribers.forEach(listener => {
            try {
                listener(connected);
            } catch (error) {
                logger.error('Error in connection status listener:', error);
            }
        });
    }

    public getCurrentSettings(): Settings {
        return JSON.parse(JSON.stringify(this.settings));
    }

    public dispose(): void {
        this.subscribers.clear();
        this.connectionSubscribers.clear();
    }
}

// Create singleton instance with WebSocket service
const webSocketService = new WebSocketService('ws://localhost:3000/wss');
export const settingsManager = new SettingsManager(webSocketService);

// Re-export Settings interface
export type { Settings };
