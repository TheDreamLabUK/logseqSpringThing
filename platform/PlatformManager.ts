import { Logger } from '../utils/logger';

const log = new Logger('PlatformManager');

export class PlatformManager {
    private static instance: PlatformManager;
    private initialized: boolean = false;

    private constructor() {
        // Private constructor for singleton
    }

    public static getInstance(): PlatformManager {
        if (!PlatformManager.instance) {
            PlatformManager.instance = new PlatformManager();
        }
        return PlatformManager.instance;
    }

    public async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        try {
            // Check for required browser features
            this.checkRequiredFeatures();

            // Initialize platform-specific features
            await this.initializePlatformFeatures();

            this.initialized = true;
            log.info('Platform initialized successfully');
        } catch (error) {
            log.error('Platform initialization failed:', error);
            throw error;
        }
    }

    private checkRequiredFeatures(): void {
        const requiredFeatures = [
            ['WebGL2', () => !!document.createElement('canvas').getContext('webgl2')],
            ['WebSocket', () => 'WebSocket' in window],
            ['WebXR', () => 'xr' in navigator],
            ['Fetch API', () => 'fetch' in window],
            ['Web Workers', () => 'Worker' in window]
        ];

        const missingFeatures = requiredFeatures
            .filter(([, check]) => !check())
            .map(([name]) => name);

        if (missingFeatures.length > 0) {
            throw new Error(`Missing required features: ${missingFeatures.join(', ')}`);
        }
    }

    private async initializePlatformFeatures(): Promise<void> {
        // Initialize WebGL context
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        if (!gl) {
            throw new Error('Failed to initialize WebGL2 context');
        }

        // Check for required WebGL extensions
        const requiredExtensions = ['EXT_color_buffer_float', 'OES_texture_float_linear'];
        for (const ext of requiredExtensions) {
            if (!gl.getExtension(ext)) {
                throw new Error(`Required WebGL extension not available: ${ext}`);
            }
        }

        // Initialize Web Workers if needed
        // Add other platform-specific initialization here
    }

    public dispose(): void {
        // Clean up platform resources
        this.initialized = false;
    }
} 