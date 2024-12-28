import { Logger } from '../utils/logger';

const log = new Logger('XRManager');

export class XRManager {
    private static instance: XRManager;
    private session: XRSession | null = null;
    private referenceSpace: XRReferenceSpace | null = null;
    private xrSupported: boolean = false;

    private constructor() {
        // Private constructor for singleton
    }

    public static getInstance(): XRManager {
        if (!XRManager.instance) {
            XRManager.instance = new XRManager();
        }
        return XRManager.instance;
    }

    public async isXRSupported(): Promise<boolean> {
        if ('xr' in navigator) {
            try {
                this.xrSupported = await navigator.xr.isSessionSupported('immersive-vr');
                return this.xrSupported;
            } catch (error) {
                log.warn('XR support check failed:', error);
                return false;
            }
        }
        return false;
    }

    public async initXRSession(): Promise<void> {
        if (!this.xrSupported) {
            throw new Error('XR not supported');
        }

        try {
            this.session = await navigator.xr.requestSession('immersive-vr', {
                requiredFeatures: ['local-floor']
            });

            this.referenceSpace = await this.session.requestReferenceSpace('local-floor');
            
            this.session.addEventListener('end', () => {
                this.session = null;
                this.referenceSpace = null;
                log.info('XR session ended');
            });

            log.info('XR session initialized');
        } catch (error) {
            log.error('Failed to initialize XR session:', error);
            throw error;
        }
    }

    public async dispose(): Promise<void> {
        if (this.session) {
            try {
                await this.session.end();
            } catch (error) {
                log.error('Error ending XR session:', error);
                throw error;
            }
        }
    }
} 