import { platformManager } from '../platform/platformManager';
import { XRSessionManager } from './xrSessionManager';
import { createLogger } from '../core/logger';

const logger = createLogger('XRInitializer');

export class XRInitializer {
    private static instance: XRInitializer | null = null;
    private xrButton: HTMLButtonElement;
    private xrSessionManager: XRSessionManager;

    private constructor(xrSessionManager: XRSessionManager) {
        this.xrSessionManager = xrSessionManager;
        this.xrButton = document.getElementById('xr-button') as HTMLButtonElement;
        if (!this.xrButton) {
            throw new Error('XR button not found');
        }
        this.setupEventListeners();
    }

    public static getInstance(xrSessionManager: XRSessionManager): XRInitializer {
        if (!XRInitializer.instance) {
            XRInitializer.instance = new XRInitializer(xrSessionManager);
        }
        return XRInitializer.instance;
    }

    private isProcessingClick = false;
    private keyboardShortcutEnabled = !platformManager.isQuest(); // Disable for Quest

    private setupEventListeners(): void {
        // Button click handler with debounce
        this.xrButton.addEventListener('click', async () => {
            if (this.isProcessingClick) return;
            this.isProcessingClick = true;
            
            try {
                await this.onXRButtonClick();
            } finally {
                // Reset after a short delay to prevent rapid clicks
                setTimeout(() => {
                    this.isProcessingClick = false;
                }, 1000);
            }
        });

        // Keyboard shortcut only for non-Quest devices
        if (this.keyboardShortcutEnabled) {
            document.addEventListener('keydown', (event) => {
                if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'a') {
                    this.onXRButtonClick();
                }
            });
        }

        // Update button visibility based on XR session state
        this.xrSessionManager.setSessionCallbacks(
            () => this.xrButton.classList.add('hidden'),    // onStart
            () => this.xrButton.classList.remove('hidden'), // onEnd
            () => {}                                        // onFrame
        );

        // Initial button state
        this.updateButtonState();
    }

    private async updateButtonState(): Promise<void> {
        const isQuest = platformManager.isQuest();
        const xrSupported = platformManager.isXRSupported();

        if (!xrSupported) {
            this.xrButton.style.display = 'none';
            return;
        }

        if (isQuest) {
            this.xrButton.textContent = 'Enter AR';
            this.xrButton.classList.remove('hidden');
        } else {
            this.xrButton.textContent = 'Enter VR';
            this.xrButton.classList.remove('hidden');
        }
    }

    private async onXRButtonClick(): Promise<void> {
        try {
            if (this.xrSessionManager.isXRPresenting()) {
                await this.xrSessionManager.endXRSession();
            } else {
                await this.xrSessionManager.initXRSession();
            }
        } catch (error) {
            logger.error('Failed to toggle XR session:', error);
        }
    }

    public dispose(): void {
        // Clean up event listeners if needed
        XRInitializer.instance = null;
    }
}