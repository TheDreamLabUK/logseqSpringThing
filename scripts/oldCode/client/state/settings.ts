import { Settings, SettingsManager as ISettingsManager, SettingCategory, SettingKey, SettingValueType } from '../types/settings';
import { createLogger } from '../utils/logger';

const logger = createLogger('SettingsManager');

type Subscriber<T extends SettingCategory, K extends SettingKey<T>> = {
    callback: (value: SettingValueType<T, K>) => void;
};

class SettingsManager implements ISettingsManager {
    private settings: Settings;
    private subscribers: Map<string, Array<Subscriber<any, any>>> = new Map();
    private initialized: boolean = false;

    constructor(defaultSettings: Settings) {
        this.settings = { ...defaultSettings };
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        const maxRetries = 3;
        const retryDelay = 1000; // 1 second

        try {
            const categories = Object.keys(this.settings) as SettingCategory[];
            
            for (const category of categories) {
                let retries = 0;
                while (retries < maxRetries) {
                    try {
                        const response = await fetch(`/api/visualization/settings/${category}`);
                        
                        if (response.ok) {
                            const data = await response.json();
                            if (this.settings[category]) {
                                this.settings[category] = { ...this.settings[category], ...data };
                                logger.info(`Loaded settings for category ${category}`);
                                break; // Success, exit retry loop
                            }
                        } else if (response.status === 404) {
                            logger.info(`Settings endpoint for ${category} not found, using defaults`);
                            break; // 404 is expected for some categories, exit retry loop
                        } else {
                            throw new Error(`Failed to fetch ${category} settings: ${response.statusText}`);
                        }
                    } catch (error) {
                        retries++;
                        if (retries === maxRetries) {
                            logger.error(`Failed to load ${category} settings after ${maxRetries} attempts:`, error);
                            logger.info(`Using default values for ${category} settings`);
                        } else {
                            logger.warn(`Retry ${retries}/${maxRetries} for ${category} settings`);
                            await new Promise(resolve => setTimeout(resolve, retryDelay));
                        }
                    }
                }
            }
            
            this.initialized = true;
            logger.info('Settings initialization complete');
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            throw error;
        }
    }

    public getCurrentSettings(): Settings {
        return this.settings;
    }

    public getDefaultSettings(): Settings {
        return this.settings;
    }

    public async updateSetting<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>
    ): Promise<void> {
        try {
            if (!(category in this.settings)) {
                throw new Error(`Invalid category: ${category}`);
            }

            const categorySettings = this.settings[category];
            if (!(String(setting) in categorySettings)) {
                throw new Error(`Invalid setting: ${String(setting)} in category ${category}`);
            }

            // Update the setting
            (this.settings[category] as any)[setting] = value;

            // Notify subscribers
            const key = `${category}.${String(setting)}`;
            const subscribers = this.subscribers.get(key) || [];
            subscribers.forEach(sub => {
                try {
                    sub.callback(value);
                } catch (error) {
                    logger.error(`Error in subscriber callback for ${key}:`, error);
                }
            });

            // Save settings to backend
            await this.saveSettings(category, setting, value);

        } catch (error) {
            logger.error(`Error updating setting ${category}.${String(setting)}:`, error);
            throw error;
        }
    }

    private async saveSettings<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        value: SettingValueType<T, K>
    ): Promise<void> {
        try {
            const response = await fetch(`/api/visualization/settings/${category}/${String(setting)}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ value }),
            });

            if (!response.ok) {
                throw new Error(`Failed to save setting: ${response.statusText}`);
            }
        } catch (error) {
            logger.error(`Error saving setting ${category}.${String(setting)}:`, error);
            throw error;
        }
    }

    public subscribe<T extends SettingCategory, K extends SettingKey<T>>(
        category: T,
        setting: K,
        callback: (value: SettingValueType<T, K>) => void
    ): () => void {
        const key = `${category}.${String(setting)}`;
        if (!this.subscribers.has(key)) {
            this.subscribers.set(key, []);
        }

        const subscriber = { callback };
        this.subscribers.get(key)!.push(subscriber);

        return () => {
            const subscribers = this.subscribers.get(key);
            if (subscribers) {
                const index = subscribers.indexOf(subscriber);
                if (index !== -1) {
                    subscribers.splice(index, 1);
                }
            }
        };
    }

    public dispose(): void {
        this.subscribers.clear();
    }
}

// Default settings that match settings.toml structure
export const defaultSettings: Settings = {
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
    enabled: true,
    enableWebsocketDebug: true,
    enableDataDebug: true,
    logBinaryHeaders: true,
    logFullJson: true
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
    port: 3001,
    rateLimitRequests: 100,
    rateLimitWindow: 60,
    tunnelId: 'dummy'
  },
  nodes: {
    baseColor: '#4CAF50',
    baseSize: 2.5,
    clearcoat: 1,
    enableHoverEffect: true,
    enableInstancing: true,
    highlightColor: '#ff4444',
    highlightDuration: 500,
    hoverScale: 1.2,
    materialType: 'phong',
    metalness: 0.5,
    opacity: 0.7,
    roughness: 0.5,
    sizeByConnections: true,
    sizeRange: [0.15, 0.4]
  },
  physics: {
    attractionStrength: 0.1,
    boundsSize: 100,
    collisionRadius: 1,
    damping: 0.8,
    enableBounds: true,
    enabled: true,
    iterations: 1,
    maxVelocity: 10,
    repulsionStrength: 0.2,
    springStrength: 0.1
  },
  rendering: {
    ambientLightIntensity: 0.5,
    backgroundColor: '#212121',
    directionalLightIntensity: 0.8,
    enableAmbientOcclusion: true,
    enableAntialiasing: true,
    enableShadows: true,
    environmentIntensity: 1
  },
  security: {
    allowedOrigins: ['*'],
    auditLogPath: '',
    cookieHttponly: true,
    cookieSamesite: 'Strict',
    cookieSecure: true,
    csrfTokenTimeout: 3600,
    enableAuditLogging: true,
    enableRequestValidation: true,
    sessionTimeout: 86400
  },
  serverDebug: {
    enabled: true,
    enableWebsocketDebug: true,
    enableDataDebug: true,
    logBinaryHeaders: true,
    logFullJson: true
  },
  websocket: {
    binaryChunkSize: 1000,
    compressionEnabled: true,
    compressionThreshold: 1024,
    heartbeatInterval: 15000,
    heartbeatTimeout: 60000,
    maxConnections: 1000,
    maxMessageSize: 1048576,
    reconnectAttempts: 3,
    reconnectDelay: 5000,
    updateRate: 60,
    url: '/wss'  // Default WebSocket endpoint
  }
};

// Re-export Settings interface
export type { Settings } from '../types/settings';

// Initialize settings from settings.toml
export const settingsManager = new SettingsManager(defaultSettings);
