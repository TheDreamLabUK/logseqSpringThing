/**
 * Platform detection and capability management
 */

import { Platform, PlatformCapabilities } from '../core/types';
import { createLogger } from '../core/utils';

const logger = createLogger('PlatformManager');

export class PlatformManager {
  private static instance: PlatformManager;
  private platform: Platform;
  private capabilities: PlatformCapabilities;

  private constructor() {
    this.platform = this.detectPlatform();
    // Initialize with default values
    this.capabilities = {
      xrSupported: false,
      webglSupported: false,
      websocketSupported: false
    };
    // Then update capabilities asynchronously
    this.updateCapabilities();
    
    logger.log(`Platform: ${this.platform}`);
  }

  static getInstance(): PlatformManager {
    if (!PlatformManager.instance) {
      PlatformManager.instance = new PlatformManager();
    }
    return PlatformManager.instance;
  }

  private detectPlatform(): Platform {
    // Check for Oculus Browser
    const userAgent = navigator.userAgent.toLowerCase();
    if (userAgent.includes('oculus') || userAgent.includes('quest')) {
      return 'quest';
    }
    return 'browser';
  }

  private async updateCapabilities(): Promise<void> {
    // Check WebXR support
    if (navigator.xr) {
      try {
        this.capabilities.xrSupported = await navigator.xr.isSessionSupported('immersive-ar');
      } catch (error) {
        logger.warn('Error checking XR support:', error);
        this.capabilities.xrSupported = false;
      }
    }

    // Check WebGL support
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      this.capabilities.webglSupported = !!gl;
    } catch (error) {
      logger.warn('Error checking WebGL support:', error);
      this.capabilities.webglSupported = false;
    }

    // Check WebSocket support
    this.capabilities.websocketSupported = 'WebSocket' in window;

    logger.log('Capabilities:', this.capabilities);
  }

  getPlatform(): Platform {
    return this.platform;
  }

  getCapabilities(): PlatformCapabilities {
    return this.capabilities;
  }

  isQuest(): boolean {
    return this.platform === 'quest';
  }

  isBrowser(): boolean {
    return this.platform === 'browser';
  }

  async isXRSupported(): Promise<boolean> {
    return this.capabilities.xrSupported;
  }

  isWebGLSupported(): boolean {
    return this.capabilities.webglSupported;
  }

  isWebSocketSupported(): boolean {
    return this.capabilities.websocketSupported;
  }

  async requestXRSession(mode: XRSessionMode = 'immersive-ar'): Promise<XRSession | null> {
    if (!this.capabilities.xrSupported || !navigator.xr) {
      logger.warn('XR not supported on this platform');
      return null;
    }

    try {
      const session = await navigator.xr.requestSession(mode, {
        requiredFeatures: ['local-floor', 'hit-test'],
        optionalFeatures: ['hand-tracking', 'layers']
      });
      return session;
    } catch (error) {
      logger.error('Error requesting XR session:', error);
      return null;
    }
  }

  // Event handling for platform-specific features
  private eventListeners: Map<string, Set<Function>> = new Map();

  on(event: string, callback: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)?.add(callback);
  }

  off(event: string, callback: Function): void {
    this.eventListeners.get(event)?.delete(callback);
  }

  private emit(event: string, ...args: any[]): void {
    this.eventListeners.get(event)?.forEach(callback => {
      try {
        callback(...args);
      } catch (error) {
        logger.error(`Error in platform event listener for ${event}:`, error);
      }
    });
  }

  // Device orientation handling for mobile/Quest
  private setupDeviceOrientation(): void {
    if (typeof DeviceOrientationEvent !== 'undefined') {
      window.addEventListener('deviceorientation', (event: DeviceOrientationEvent) => {
        this.emit('orientation', {
          alpha: event.alpha, // z-axis rotation
          beta: event.beta,   // x-axis rotation
          gamma: event.gamma  // y-axis rotation
        });
      }, true);
    }
  }

  // Screen orientation handling
  private setupScreenOrientation(): void {
    if ('screen' in window && 'orientation' in screen) {
      screen.orientation.addEventListener('change', () => {
        this.emit('orientationchange', screen.orientation.type);
      });
    }
  }

  // Initialize platform-specific features
  async initialize(): Promise<void> {
    // Set up event listeners
    this.setupDeviceOrientation();
    this.setupScreenOrientation();

    // Check for WebXR changes
    if (navigator.xr) {
      navigator.xr.addEventListener('devicechange', async () => {
        if (navigator.xr) {
          this.capabilities.xrSupported = await navigator.xr.isSessionSupported('immersive-ar');
          this.emit('xrdevicechange', this.capabilities.xrSupported);
        }
      });
    }

    // Additional platform-specific initialization can be added here
    logger.log('Platform manager initialized');
  }
}

// Export a singleton instance
export const platformManager = PlatformManager.getInstance();
