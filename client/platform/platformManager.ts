import { Platform, PlatformCapabilities } from '../core/types';
import { createLogger } from '../core/utils';
import { Settings } from '../types/settings';
import { XRSessionMode } from '../types/xr';
import { WebGLRenderer } from 'three';

const logger = createLogger('PlatformManager');

declare global {
  interface Navigator {
    xr?: XRSystem;
  }
}

interface PlatformFeatures {
  xr?: {
    isSupported: boolean;
    isImmersiveSupported: boolean;
  };
  webgl?: {
    isSupported: boolean;
    version: number;
  };
  handTracking: boolean;
  planeDetection: boolean;
  requestAnimationFrame: (callback: FrameRequestCallback) => number;
  cancelAnimationFrame: (handle: number) => void;
  getWebGLContext: () => WebGLRenderingContext | null;
}

interface PlatformCapabilities {
  webgl: boolean;
  webgl2: boolean;
  webxr: boolean;
  handTracking: boolean;
  planeDetection: boolean;
}

interface EventListener {
  (...args: unknown[]): void;
}

interface EventMap {
  [event: string]: EventListener[];
}

class BrowserEventEmitter {
  private listeners: EventMap = {};

  on(event: string, listener: EventListener): void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(listener);
  }

  emit(event: string, ...args: unknown[]): void {
    const eventListeners = this.listeners[event];
    if (eventListeners) {
      eventListeners.forEach(listener => listener(...args));
    }
  }

  removeAllListeners(): void {
    this.listeners = {};
  }
}

export class PlatformManager extends BrowserEventEmitter {
  private static instance: PlatformManager | null = null;
  private platform: Platform;
  private features: PlatformFeatures;
  private initialized: boolean = false;
  private renderer: WebGLRenderer | null = null;
  private _settings: Settings;

  private constructor(settings: Settings) {
    super();
    this._settings = settings;
    this.platform = 'desktop';
    this.features = this.detectFeatures();
  }

  static getInstance(settings: Settings): PlatformManager {
    if (!PlatformManager.instance) {
      PlatformManager.instance = new PlatformManager(settings);
    }
    return PlatformManager.instance;
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    this.detectPlatform();
    await this.detectCapabilities();
    
    // Initialize platform with settings
    if (this._settings.xr?.mode) {
      this.features.xr = await this.checkXRSupport(this._settings.xr.mode as XRSessionMode);
    }
    
    this.initialized = true;
    logger.log('Platform manager initialized');
  }

  private detectPlatform(): void {
    const userAgent = navigator.userAgent.toLowerCase();
    const isQuest = userAgent.includes('quest');
    
    if (isQuest) {
      this.platform = 'quest';
    } else if (userAgent.includes('chrome') || userAgent.includes('firefox') || userAgent.includes('safari')) {
      this.platform = 'browser';
    } else {
      this.platform = 'desktop';
    }
  }

  private async detectCapabilities(): Promise<void> {
    // WebXR support
    if ('xr' in navigator && navigator.xr) {
      try {
        this.features.xr = {
          isSupported: await navigator.xr.isSessionSupported('immersive-ar'),
          isImmersiveSupported: await navigator.xr.isSessionSupported('immersive-ar')
        };
        this.features.handTracking = this.features.xr.isSupported;
        this.features.planeDetection = this.features.xr.isSupported;
      } catch (error) {
        logger.warn('WebXR not supported:', error);
        this.features.xr = {
          isSupported: false,
          isImmersiveSupported: false
        };
        this.features.handTracking = false;
        this.features.planeDetection = false;
      }
    }

    logger.log('Platform capabilities detected:', this.getCapabilities());
  }

  private detectFeatures(): PlatformFeatures {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

    return {
      xr: {
        isSupported: 'xr' in navigator,
        isImmersiveSupported: 'xr' in navigator
      },
      webgl: {
        isSupported: !!gl,
        version: gl ? (canvas.getContext('webgl2') ? 2 : 1) : 0
      },
      handTracking: false, // Implement proper detection
      planeDetection: false, // Implement proper detection
      requestAnimationFrame: window.requestAnimationFrame.bind(window),
      cancelAnimationFrame: window.cancelAnimationFrame.bind(window),
      getWebGLContext: () => gl
    };
  }

  getPlatform(): Platform {
    return this.platform;
  }

  getCapabilities(): PlatformCapabilities {
    return {
      webgl: this.features.webgl.isSupported,
      webgl2: this.features.webgl.version === 2,
      webxr: this.features.xr.isSupported,
      handTracking: this.features.handTracking,
      planeDetection: this.features.planeDetection
    };
  }

  isDesktop(): boolean {
    return this.platform === 'desktop';
  }

  isQuest(): boolean {
    return this.platform === 'quest';
  }

  isBrowser(): boolean {
    return this.platform === 'browser';
  }

  isXRSupported(): boolean {
    return this.features.xr.isSupported;
  }

  isWebGLSupported(): boolean {
    return this.features.webgl.isSupported;
  }

  isWebSocketSupported(): boolean {
    return 'WebSocket' in window;
  }

  async requestXRSession(): Promise<XRSession | null> {
    if (!this.features.xr.isSupported || !('xr' in navigator) || !navigator.xr) {
      logger.warn('WebXR not supported');
      return null;
    }

    try {
      const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor', 'hit-test'],
        optionalFeatures: ['hand-tracking', 'plane-detection']
      });

      // Update capabilities based on session features
      session.addEventListener('end', () => {
        logger.log('XR session ended');
        this.emit('xrsessionend');
      });

      logger.log('XR session started');
      return session;
    } catch (error) {
      logger.error('Failed to start XR session:', error);
      return null;
    }
  }

  async checkXRSupport(mode: XRSessionMode = 'immersive-ar'): Promise<boolean> {
    if ('xr' in navigator && navigator.xr) {
      try {
        const supported = await navigator.xr.isSessionSupported(mode);
        if (supported) {
          this.features.xr.isSupported = true;
          this.features.handTracking = true;
          this.features.planeDetection = true;
          this.emit('xrdevicechange', true);
          logger.log('WebXR supported for mode:', mode);
          return true;
        }
      } catch (error) {
        logger.warn('WebXR check failed:', error);
      }
    }
    this.features.xr.isSupported = false;
    this.features.handTracking = false;
    this.features.planeDetection = false;
    this.emit('xrdevicechange', false);
    return false;
  }

  setRenderer(renderer: WebGLRenderer): void {
    this.renderer = renderer;
  }

  getRenderer(): WebGLRenderer | null {
    return this.renderer;
  }

  requestAnimationFrame(callback: FrameRequestCallback): number {
    return this.features.requestAnimationFrame(callback);
  }

  cancelAnimationFrame(handle: number): void {
    this.features.cancelAnimationFrame(handle);
  }

  getWebGLContext(): WebGLRenderingContext | null {
    return this.features.getWebGLContext();
  }

  dispose(): void {
    this.removeAllListeners();
    this.initialized = false;
    PlatformManager.instance = null;
  }
}

export const platformManager = PlatformManager.getInstance({} as Settings);
