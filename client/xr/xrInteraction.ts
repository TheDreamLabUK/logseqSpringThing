import * as THREE from 'three';
import { NodeManager } from '../rendering/nodes';
import { XRSessionManager } from './xrSessionManager';
import { Platform, Node } from '../core/types';
import { Settings } from '../types/settings';
import { XRHandWithHaptics } from '../types/xr';
import { platformManager } from '../platform/platformManager';
import { createLogger } from '../core/logger';
import { SettingsStore } from '../state/SettingsStore';
import { defaultSettings } from '../state/defaultSettings';
import { HandGestureType } from '../types/gestures';
import { WebSocketService } from '../websocket/websocketService';

const logger = createLogger('XRInteraction');

interface HapticActuator {
    pulse: (intensity: number, duration: number) => Promise<boolean>;
}

interface WorldObject3D extends THREE.Object3D {
    getWorldPosition(target: THREE.Vector3): THREE.Vector3;
}

export class XRInteraction {
    private static instance: XRInteraction | null = null;
    private xrManager: XRSessionManager;
    private nodeManager: NodeManager;
    private controllers: THREE.Group[] = [];
    private lastInteractorPosition = new THREE.Vector3();
    private hands: XRHandWithHaptics[] = [];
    private settings: Settings;
    private settingsStore: SettingsStore;
    private selectedNodeId: string | null = null;
    private worldPosition = new THREE.Vector3();
    private websocketService: WebSocketService;
    private handGestureStates: Map<number, HandGestureType> = new Map();
    private updateBatch: Map<string, THREE.Vector3> = new Map();
    private batchUpdateTimeout: NodeJS.Timeout | null = null;

    private constructor(xrManager: XRSessionManager, nodeManager: NodeManager) {
        this.xrManager = xrManager;
        this.nodeManager = nodeManager;
        this.settingsStore = SettingsStore.getInstance();
        this.settings = defaultSettings;
        this.websocketService = WebSocketService.getInstance();
        
        this.setupXRControllers();
        this.setupHandTracking();
        this.setupPlatformListeners();
        this.setupSettingsSubscription();
    }

    private setupPlatformListeners(): void {
        platformManager.on('platformChange', (platform: Platform) => {
            logger.info(`Platform changed to ${platform}`);
            this.updateXRFeatures();
        });
    }

    private updateXRFeatures(): void {
        const platform = platformManager.getPlatform();
        const capabilities = platformManager.getCapabilities();

        // Update hand tracking based on platform capabilities
        if (capabilities.handTracking) {
            this.setupHandTracking();
        } else {
            this.disableHandTracking();
        }

        // Update haptics based on platform capabilities
        this.controllers.forEach(controller => {
            if (controller.userData) {
                controller.userData.platform = platform;
            }
        });

        this.hands.forEach(hand => {
            if (hand.userData) {
                hand.userData.platform = platform;
            }
        });
    }

    private disableHandTracking(): void {
        this.hands.forEach(hand => {
            if (hand.parent) {
                hand.parent.remove(hand);
            }
        });
        this.hands = [];
    }

    private setupSettingsSubscription(): void {
        // Subscribe to all XR input settings changes
        const inputSettings = [
            'enableHandTracking',
            'enableHaptics',
            'hapticIntensity',
            'dragThreshold',
            'pinchThreshold',
            'rotationThreshold',
            'interactionRadius'
        ];

        inputSettings.forEach(setting => {
            this.settingsStore.subscribe(`xr.input.${setting}`, (value) => {
                if (this.settings.xr) {
                    this.settings = {
                        ...this.settings,
                        xr: {
                            ...this.settings.xr,
                            input: {
                                ...this.settings.xr.input,
                                [setting]: value
                            }
                        }
                    };
                }
                this.updateXRFeatures();
            });
        });
    }

    private setupXRControllers(): void {
        this.xrManager.onControllerAdded((controller: THREE.Group) => {
            controller.userData.platform = platformManager.getPlatform();
            this.controllers.push(controller);
            if (controller.userData.hapticActuator && this.settings.xr.input.enableHaptics) {
                this.triggerHapticFeedback(controller, 0.5, 50);
            }
        });

        this.xrManager.onControllerRemoved((controller: THREE.Group) => {
            const index = this.controllers.indexOf(controller);
            if (index !== -1) {
                this.controllers.splice(index, 1);
            }
        });
    }

    private setupHandTracking(): void {
        if (!platformManager.getCapabilities().handTracking) {
            logger.info('Hand tracking not supported on this platform');
            return;
        }
        // Hand tracking is now handled internally
        // No need to register with XRSessionManager
    }

    private flushPositionUpdates(): void {
        if (this.updateBatch.size === 0) return;

        const updates = Array.from(this.updateBatch.entries()).map(([id, position]) => ({
            id,
            position: { x: position.x, y: position.y, z: position.z }
        }));

        // Update each node position individually
        updates.forEach(update => {
            const newPosition = new THREE.Vector3(update.position.x, update.position.y, update.position.z);
            this.nodeManager.updateNodePosition(update.id, newPosition);
        });

        this.websocketService.sendNodeUpdates(updates);
        this.updateBatch.clear();
    }

    private queuePositionUpdate(nodeId: string, position: THREE.Vector3): void {
        this.updateBatch.set(nodeId, position.clone());
        
        if (this.batchUpdateTimeout) {
            clearTimeout(this.batchUpdateTimeout);
        }

        this.batchUpdateTimeout = setTimeout(() => {
            this.flushPositionUpdates();
            this.batchUpdateTimeout = null;
        }, 16); // ~60fps
    }

    private async triggerHapticFeedback(controller: THREE.Group, intensity: number, duration: number): Promise<void> {
        if (!this.settings.xr.input.enableHaptics) return;

        const hapticActuator = controller.userData.hapticActuator as HapticActuator;
        if (hapticActuator) {
            try {
                await hapticActuator.pulse(
                    intensity * this.settings.xr.input.hapticIntensity,
                    duration
                );
            } catch (error) {
                logger.warn('Failed to trigger haptic feedback:', error);
            }
        }
    }

    public static getInstance(xrManager: XRSessionManager, nodeManager: NodeManager): XRInteraction {
        if (!XRInteraction.instance) {
            XRInteraction.instance = new XRInteraction(xrManager, nodeManager);
        }
        return XRInteraction.instance;
    }

    public update(): void {
        if (!this.settings.xr.input.enableHandTracking && !this.controllers.length) {
            return;
        }

        // Update hand interactions
        this.hands.forEach(hand => {
            if (hand.pinchStrength > this.settings.xr.input.pinchThreshold) {
                this.handlePinchGesture(hand);
            }
        });

        // Update controller interactions
        this.controllers.forEach(controller => {
            if (controller.userData.isSelecting) {
                this.handleControllerInteraction(controller);
            }
        });
    }

    private handlePinchGesture(hand: XRHandWithHaptics): void {
        const indexTip = hand.hand.joints['index-finger-tip'];
        if (!indexTip) return;

        try {
            (indexTip as WorldObject3D).getWorldPosition(this.worldPosition);

            if (this.lastInteractorPosition.distanceTo(this.worldPosition) > this.settings.xr.input.dragThreshold) {
                if (!this.selectedNodeId) {
                    this.selectedNodeId = this.findClosestNode(this.worldPosition);
                }

                if (this.selectedNodeId) {
                    this.queuePositionUpdate(this.selectedNodeId, this.worldPosition);
                    this.lastInteractorPosition.copy(this.worldPosition);

                    if (hand.userData.hapticActuator && this.settings.xr.input.enableHaptics) {
                        this.triggerHapticFeedback(hand, 0.3, 30);
                    }
                }
            }
        } catch (error) {
            logger.error('Error handling pinch gesture:', error);
        }
    }

    private handleControllerInteraction(controller: THREE.Group): void {
        try {
            (controller as WorldObject3D).getWorldPosition(this.worldPosition);

            if (this.lastInteractorPosition.distanceTo(this.worldPosition) > this.settings.xr.input.dragThreshold) {
                if (!this.selectedNodeId) {
                    this.selectedNodeId = this.findClosestNode(this.worldPosition);
                }

                if (this.selectedNodeId) {
                    this.queuePositionUpdate(this.selectedNodeId, this.worldPosition);
                    this.lastInteractorPosition.copy(this.worldPosition);

                    if (controller.userData.hapticActuator && this.settings.xr.input.enableHaptics) {
                        this.triggerHapticFeedback(controller, 0.3, 30);
                    }
                }
            }
        } catch (error) {
            logger.error('Error handling controller interaction:', error);
        }
    }

    private findClosestNode(position: THREE.Vector3): string | null {
        const nodes = this.nodeManager.getCurrentNodes() as Array<Node>;
        let closestNode: Node | null = null;
        let closestDistance = Infinity;

        for (const node of nodes as Array<Node>) {
            if (!node || !node.data || !node.data.position) continue;
            
            const nodePos = new THREE.Vector3(
                node.data.position.x,
                node.data.position.y,
                node.data.position.z
            );
            const distance = position.distanceTo(nodePos);
            if (distance < closestDistance && distance < (this.settings.xr.input.interactionRadius || 0.5)) {
                closestDistance = distance;
                closestNode = node;
            }
        }

        return closestNode?.id || null;
    }

    public dispose(): void {
        if (this.batchUpdateTimeout) {
            clearTimeout(this.batchUpdateTimeout);
            this.batchUpdateTimeout = null;
        }

        // Flush any remaining updates
        this.flushPositionUpdates();

        // Clear subscriptions
        // Unsubscribe from all XR input settings
        [
            'enableHandTracking',
            'enableHaptics',
            'hapticIntensity',
            'dragThreshold',
            'pinchThreshold',
            'rotationThreshold',
            'interactionRadius'
        ].forEach(setting => {
            this.settingsStore.unsubscribe(`xr.input.${setting}`, () => {});
        });

        // Clear data structures
        this.controllers = [];
        this.hands = [];
        this.handGestureStates.clear();
        this.updateBatch.clear();
        this.lastInteractorPosition.set(0, 0, 0);
        this.selectedNodeId = null;
        XRInteraction.instance = null;
    }
}
