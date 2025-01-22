import { Platform, PlatformCapabilities } from '../core/types';
import { createLogger } from '../core/utils';
import { Settings } from '../types/settings';
import { XRSessionMode } from '../types/xr';

const logger = createLogger('PlatformManager');

declare global {
  interface Navigator {
    xr?: XRSystem;
  }
}

class BrowserEventEmitter {
  private listeners: { [event: string]: Function[] } = {};

  on(event: string, listener: Function): void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(listener);
  }

  emit(event: string, ...args: any[]): void {
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
  private capabilities: PlatformCapabilities;
  private initialized: boolean = false;
  private _isXRMode: boolean = false;

  private constructor() {
    super();
    this.platform = 'desktop';
    this.capabilities = {
      xrSupported: false,
      webglSupported: false,
      websocketSupported: false,
      webxr: false,
      handTracking: false,
      planeDetection: false
    };
  }

  static getInstance(): PlatformManager {
    if (!PlatformManager.instance) {
      PlatformManager.instance = new PlatformManager();
    }
    return PlatformManager.instance;
  }

  async initialize(settings: Settings): Promise<void> {
    if (this.initialized) {
      return;
    }

    this.detectPlatform();
    await this.detectCapabilities();
    
    // Initialize platform with settings
    if (settings.xr?.mode) {
      this._isXRMode = true;
      this.capabilities.xrSupported = await this.checkXRSupport(
        settings.xr?.mode as XRSessionMode
      );
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
        this.capabilities.xrSupported = await navigator.xr.isSessionSupported('immersive-ar');
        this.capabilities.webxr = this.capabilities.xrSupported;
        this.capabilities.handTracking = this.capabilities.xrSupported;
        this.capabilities.planeDetection = this.capabilities.xrSupported;
      } catch (error) {
        logger.warn('WebXR not supported:', error);
        this.capabilities.xrSupported = false;
        this.capabilities.webxr = false;
        this.capabilities.handTracking = false;
        this.capabilities.planeDetection = false;
      }
    }

    // WebGL support
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      this.capabilities.webglSupported = !!gl;
    } catch (error) {
      logger.warn('WebGL not supported:', error);
      this.capabilities.webglSupported = false;
    }

    // WebSocket support
    this.capabilities.websocketSupported = 'WebSocket' in window;

    logger.log('Platform capabilities detected:', this.capabilities);
  }

  getPlatform(): Platform {
    return this.platform;
  }

  getCapabilities(): PlatformCapabilities {
    return { ...this.capabilities };
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
    return this.capabilities.xrSupported;
  }

  isWebGLSupported(): boolean {
    return this.capabilities.webglSupported;
  }

  isWebSocketSupported(): boolean {
    return this.capabilities.websocketSupported;
  }

  async requestXRSession(): Promise<XRSession | null> {
    if (!this.capabilities.xrSupported || !('xr' in navigator) || !navigator.xr) {
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
          this.capabilities.webxr = true;
          this.capabilities.handTracking = true;
          this.capabilities.planeDetection = true;
          this.emit('xrdevicechange', true);
          logger.log('WebXR supported for mode:', mode);
          return true;
        }
      } catch (error) {
        logger.warn('WebXR check failed:', error);
      }
    }
    this.capabilities.webxr = false;
    this.capabilities.handTracking = false;
    this.capabilities.planeDetection = false;
    this.emit('xrdevicechange', false);
    return false;
  }

  dispose(): void {
    this.removeAllListeners();
    this.initialized = false;
    PlatformManager.instance = null;
  }

  get isXRMode(): boolean {
    return this._isXRMode;
  }

  setXRMode(enabled: boolean): void {
    this._isXRMode = enabled;
    this.emit('xrmodechange', enabled);
  }
}

export const platformManager = PlatformManager.getInstance();
