import { Logger } from './utils/logger';
import { LoadingOverlay } from './ui/LoadingOverlay';
import { ErrorDisplay } from './ui/ErrorDisplay';
import { ControlPanel } from './ui/ControlPanel';
import { WebSocketService } from './websocket/websocketService';
import { SettingsStore } from './state/SettingsStore';
import { XRManager } from './xr/XRManager';
import { PlatformManager } from './platform/PlatformManager';

const log = new Logger('Application');

interface EventListener {
    element: HTMLElement;
    type: string;
    handler: (event: Event) => void;
}

export class Application {
    private loadingOverlay: LoadingOverlay;
    private errorDisplay: ErrorDisplay;
    private controlPanel: ControlPanel;
    private settingsStore: SettingsStore;
    private webSocketService: WebSocketService;
    private xrManager: XRManager;
    private platformManager: PlatformManager;
    private eventListeners: EventListener[] = [];
    private connectionStatusElement: HTMLElement | null = null;

    constructor() {
        this.loadingOverlay = new LoadingOverlay();
        this.errorDisplay = new ErrorDisplay();
        this.controlPanel = new ControlPanel();
        this.settingsStore = new SettingsStore();
        this.webSocketService = WebSocketService.getInstance();
        this.xrManager = XRManager.getInstance();
        this.platformManager = PlatformManager.getInstance();
    }

    public async initialize(): Promise<void> {
        try {
            this.loadingOverlay.show();

            // Initialize platform
            await this.initializePlatform();

            // Initialize settings
            await this.settingsStore.retryInitialize();

            // Initialize WebSocket
            await this.webSocketService.initialize();

            // Initialize XR if available
            await this.initializeXR();

            // Setup UI
            this.setupUI();

            // Hide loading overlay
            await this.loadingOverlay.hide();
        } catch (error) {
            log.error('Failed to initialize application:', error);
            this.errorDisplay.show(`Failed to initialize application: ${error.message}`);
            throw error;
        }
    }

    private async initializePlatform(): Promise<void> {
        try {
            await this.platformManager.initialize();
        } catch (error) {
            log.error('Failed to initialize platform:', error);
            throw new Error(`Platform initialization failed: ${error.message}`);
        }
    }

    private async initializeXR(): Promise<void> {
        try {
            if (await this.xrManager.isXRSupported()) {
                await this.xrManager.initXRSession();
            }
        } catch (error) {
            log.warn('Failed to initialize XR:', error);
            this.errorDisplay.show('XR initialization failed. Falling back to non-XR mode.', 5000);
            // Continue without XR
        }
    }

    private setupUI(): void {
        this.setupConnectionStatus();
        this.setupUIEventListeners();
    }

    private setupConnectionStatus(): void {
        this.connectionStatusElement = document.getElementById('connection-status');
        if (!this.connectionStatusElement) {
            log.warn('Connection status element not found');
            return;
        }

        this.webSocketService.setConnectionStatusHandler(status => {
            this.updateConnectionStatus(status);
        });
    }

    private updateConnectionStatus(status: ConnectionState): void {
        if (!this.connectionStatusElement) {
            return;
        }

        const statusText = ConnectionState[status];
        const statusClass = status === ConnectionState.Connected ? 'connected' : 'disconnected';

        this.connectionStatusElement.textContent = statusText;
        this.connectionStatusElement.className = `connection-status ${statusClass}`;
    }

    private setupUIEventListeners(): void {
        const addListener = (element: HTMLElement, type: string, handler: (event: Event) => void) => {
            element.addEventListener(type, handler);
            this.eventListeners.push({ element, type, handler });
        };

        // Add your event listeners here using addListener
        addListener(this.controlPanel.getElement(), 'save', () => this.controlPanel.saveSettings());
    }

    public async dispose(): Promise<void> {
        const components = [
            { name: 'WebSocket', dispose: () => this.webSocketService.dispose() },
            { name: 'XR', dispose: () => this.xrManager.dispose() },
            { name: 'Settings', dispose: () => this.settingsStore.dispose() },
            { name: 'ControlPanel', dispose: () => this.controlPanel.dispose() },
            { name: 'ErrorDisplay', dispose: () => this.errorDisplay.dispose() },
            { name: 'LoadingOverlay', dispose: () => this.loadingOverlay.dispose() }
        ];

        const errors: Error[] = [];

        // Remove event listeners
        this.eventListeners.forEach(({ element, type, handler }) => {
            try {
                element.removeEventListener(type, handler);
            } catch (error) {
                log.error('Failed to remove event listener:', error);
                errors.push(error as Error);
            }
        });
        this.eventListeners = [];

        // Dispose components
        for (const component of components) {
            try {
                await component.dispose();
            } catch (error) {
                log.error(`Failed to dispose ${component.name}:`, error);
                errors.push(error as Error);
            }
        }

        if (errors.length > 0) {
            throw new Error(`Failed to dispose some components: ${errors.map(e => e.message).join(', ')}`);
        }
    }
} 