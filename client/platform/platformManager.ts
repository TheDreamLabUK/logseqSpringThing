import { Platform, PlatformCapabilities as IPlatformCapabilities } from '../core/types';
import { Settings } from '../types/settings';
import { createLogger } from '../core/logger';
import { XRSessionMode } from '../types/xr';

const logger = createLogger('PlatformManager');

interface PlatformFeatures {
    webgl?: {
        isSupported: boolean;
        version: number;
    };
    xr?: {
        isSupported: boolean;
        isImmersiveSupported: boolean;
    };
}

export class PlatformManager {
    private static instance: PlatformManager | null = null;
    private initialized = false;
    private platform: Platform | null = null;
    private features: PlatformFeatures = {};
    private _settings: Settings;

    private constructor(settings: Settings) {
        this._settings = settings;
    }

    static getInstance(settings?: Settings): PlatformManager {
        if (!PlatformManager.instance && settings) {
            PlatformManager.instance = new PlatformManager(settings);
        }
        return PlatformManager.instance!;
    }

    async init(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            await this.detectPlatform();
            await this.detectFeatures();
            this.initialized = true;
            logger.info('Platform manager initialized');
        } catch (error) {
            logger.error('Failed to initialize platform manager:', error);
            throw error;
        }
    }

    private async detectPlatform(): Promise<void> {
        // Default to desktop
        this.platform = {
            name: 'desktop',
            version: '1.0.0',
            capabilities: {
                webgl: { isSupported: false, version: 0 },
                xr: { isSupported: false, isImmersiveSupported: false }
            }
        };

        // Detect platform type
        if (navigator.userAgent.includes('Quest')) {
            this.platform.name = 'quest';
        } else if (navigator.userAgent.includes('Mobile')) {
            this.platform.name = 'mobile';
        } else if (navigator.userAgent.includes('Firefox') || navigator.userAgent.includes('Chrome')) {
            this.platform.name = 'browser';
        }
    }

    private async detectFeatures(): Promise<void> {
        // Detect WebGL support
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        this.features.webgl = {
            isSupported: !!gl,
            version: gl?.getParameter(gl.VERSION) ? 2 : 1
        };

        // Detect XR support
        this.features.xr = {
            isSupported: false,
            isImmersiveSupported: false
        };

        if ('xr' in navigator) {
            this.features.xr.isSupported = true;
            this.features.xr.isImmersiveSupported = await this.checkXRSupport(this._settings.xr.mode as XRSessionMode);
        }

        logger.info('Platform capabilities detected:', this.getCapabilities());
    }

    getCapabilities(): IPlatformCapabilities {
        return {
            webgl: {
                isSupported: this.features.webgl?.isSupported || false,
                version: this.features.webgl?.version || 0
            },
            xr: {
                isSupported: this.features.xr?.isSupported || false,
                isImmersiveSupported: this.features.xr?.isImmersiveSupported || false
            }
        };
    }

    isDesktop(): boolean {
        return this.platform?.name === 'desktop';
    }

    isQuest(): boolean {
        return this.platform?.name === 'quest';
    }

    isBrowser(): boolean {
        return this.platform?.name === 'browser';
    }

    supportsXR(): boolean {
        return this.features.xr?.isSupported || false;
    }

    supportsWebGL(): boolean {
        return this.features.webgl?.isSupported || false;
    }

    private async checkXRSupport(mode: XRSessionMode): Promise<boolean> {
        if (!this.features.xr?.isSupported || !('xr' in navigator) || !navigator.xr) {
            return false;
        }

        try {
            return await navigator.xr.isSessionSupported(mode);
        } catch (error) {
            logger.warn('Failed to check XR support:', error);
            return false;
        }
    }

    async startXRSession(): Promise<XRSession | null> {
        if (!this.supportsXR() || !navigator.xr) {
            return null;
        }

        try {
            const session = await navigator.xr.requestSession(this._settings.xr.mode as XRSessionMode);
            session.addEventListener('end', () => {
                logger.info('XR session ended');
            });
            logger.info('XR session started');
            return session;
        } catch (error) {
            logger.error('Failed to start XR session:', error);
            return null;
        }
    }

    dispose(): void {
        this.initialized = false;
        PlatformManager.instance = null;
    }
}

export const platformManager = PlatformManager.getInstance();
