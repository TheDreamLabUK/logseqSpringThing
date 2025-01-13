import { XRSessionManager } from './xrSessionManager';
import { NodeManager } from '../rendering/nodes';
import { SettingsStore } from '../state/SettingsStore';
import { createLogger } from '../core/logger';
import { WebSocketService } from '../websocket/websocketService';
import * as THREE from 'three';

const logger = createLogger('XRInteraction');

export class XRInteraction {
    private static instance: XRInteraction | null = null;
    private readonly settingsStore: SettingsStore;
    private updateBatch: Map<string, THREE.Vector3> = new Map();
    private batchUpdateTimeout: NodeJS.Timeout | null = null;
    private settingsUnsubscribers: Array<() => void> = [];
    private interactionEnabled: boolean = false;
    private websocketService: WebSocketService;

    private constructor(_: XRSessionManager, __: NodeManager) {
        this.settingsStore = SettingsStore.getInstance();
        this.websocketService = WebSocketService.getInstance();
        this.initializeSettings();
    }

    private initializeSettings(): void {
        try {
            this.setupSettingsSubscription();
        } catch (error) {
            logger.error('Failed to setup settings subscription:', error);
        }
    }

    public static getInstance(xrManager: XRSessionManager, nodeManager: NodeManager): XRInteraction {
        if (!XRInteraction.instance) {
            XRInteraction.instance = new XRInteraction(xrManager, nodeManager);
        }
        return XRInteraction.instance;
    }

    private setupSettingsSubscription(): void {
        // Clear any existing subscriptions
        this.settingsUnsubscribers.forEach(unsub => unsub());
        this.settingsUnsubscribers = [];

        // Subscribe to XR interaction enabled state
        let unsubscriber: (() => void) | undefined;
        this.settingsStore.subscribe('xr.interaction.enabled', (value) => {
            this.interactionEnabled = typeof value === 'boolean' ? value : value === 'true';
            if (!this.interactionEnabled) {
                this.clearHandState();
            }
        }).then(unsub => {
            unsubscriber = unsub;
            if (unsubscriber) {
                this.settingsUnsubscribers.push(unsubscriber);
            }
        });
    }

    private clearHandState(): void {
        this.updateBatch.clear();
        if (this.batchUpdateTimeout) {
            clearTimeout(this.batchUpdateTimeout);
            this.batchUpdateTimeout = null;
        }
    }

    private flushPositionUpdates(): void {
        if (this.updateBatch.size === 0) return;

        const updates = Array.from(this.updateBatch.entries()).map(([id, position]) => ({
            id,
            position: {
                x: position.x,
                y: position.y,
                z: position.z
            }
        }));

        this.websocketService.sendNodeUpdates(updates);
        this.updateBatch.clear();
    }

    public update(): void {
        if (!this.interactionEnabled) return;
        this.flushPositionUpdates();
    }

    public dispose(): void {
        // Clear subscriptions
        this.settingsUnsubscribers.forEach(unsub => unsub());
        this.settingsUnsubscribers = [];

        // Flush any pending updates
        this.flushPositionUpdates();

        XRInteraction.instance = null;
    }
}
