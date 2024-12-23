import * as THREE from 'three';
import { NodeManager } from '../rendering/nodes';
import { XRSessionManager } from './xrSessionManager';
import { Settings, Platform, Node } from '../core/types';
import { SettingsManager } from '../state/settings';
import { XRHandWithHaptics } from '../types/xr';
import { platformManager } from '../platform/platformManager';
import { createLogger } from '../core/logger';
import { SettingsStore } from '../state/SettingsStore';
import { defaultSettings } from '../state/defaultSettings';

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

    private constructor(xrManager: XRSessionManager, nodeManager: NodeManager, settingsManager: SettingsManager) {
        this.xrManager = xrManager;
        this.nodeManager = nodeManager;
        this.settingsStore = SettingsStore.getInstance();
        this.settings = defaultSettings;
        
        this.setupXRControllers();
        this.setupHandTracking();
        this.setupPlatformListeners();
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

    public static getInstance(xrManager: XRSessionManager, nodeManager: NodeManager, settingsManager: SettingsManager): XRInteraction {
        if (!XRInteraction.instance) {
            XRInteraction.instance = new XRInteraction(xrManager, nodeManager, settingsManager);
        }
        return XRInteraction.instance;
    }

    private setupXRControllers(): void {
        this.xrManager.onControllerAdded((controller: THREE.Group) => {
            controller.userData.platform = platformManager.getPlatform();
            this.controllers.push(controller);
            if (controller.userData.hapticActuator && this.settings.ar.enableHaptics) {
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

        this.xrManager.onHandAdded((hand: XRHandWithHaptics) => {
            hand.userData.platform = platformManager.getPlatform();
            this.hands.push(hand);
        });

        this.xrManager.onHandRemoved((hand: XRHandWithHaptics) => {
            const index = this.hands.indexOf(hand);
            if (index !== -1) {
                this.hands.splice(index, 1);
            }
        });
    }

    private async triggerHapticFeedback(controller: THREE.Group, intensity: number, duration: number): Promise<void> {
        if (!this.settings.ar.enableHaptics) return;

        const hapticActuator = controller.userData.hapticActuator as HapticActuator;
        if (hapticActuator) {
            try {
                await hapticActuator.pulse(
                    intensity * this.settings.ar.hapticIntensity,
                    duration
                );
            } catch (error) {
                logger.warn('Failed to trigger haptic feedback:', error);
            }
        }
    }

    public update(): void {
        if (!this.settings.ar.enableHandTracking && !this.controllers.length) {
            return;
        }

        // Update hand interactions
        this.hands.forEach(hand => {
            if (hand.pinchStrength > this.settings.ar.pinchThreshold) {
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

        (indexTip as WorldObject3D).getWorldPosition(this.worldPosition);

        if (this.lastInteractorPosition.distanceTo(this.worldPosition) > this.settings.ar.dragThreshold) {
            // Find closest node if none selected
            if (!this.selectedNodeId) {
                this.selectedNodeId = this.findClosestNode(this.worldPosition);
            }

            // Update selected node position
            if (this.selectedNodeId) {
                this.nodeManager.updateNodePosition(this.selectedNodeId, this.worldPosition);
                this.lastInteractorPosition.copy(this.worldPosition);

                // Trigger haptic feedback if available
                if (hand.userData.hapticActuator && this.settings.ar.enableHaptics) {
                    this.triggerHapticFeedback(hand, 0.3, 30);
                }
            }
        }
    }

    private handleControllerInteraction(controller: THREE.Group): void {
        (controller as WorldObject3D).getWorldPosition(this.worldPosition);

        if (this.lastInteractorPosition.distanceTo(this.worldPosition) > this.settings.ar.dragThreshold) {
            // Find closest node if none selected
            if (!this.selectedNodeId) {
                this.selectedNodeId = this.findClosestNode(this.worldPosition);
            }

            // Update selected node position
            if (this.selectedNodeId) {
                this.nodeManager.updateNodePosition(this.selectedNodeId, this.worldPosition);
                this.lastInteractorPosition.copy(this.worldPosition);

                // Trigger haptic feedback
                if (controller.userData.hapticActuator && this.settings.ar.enableHaptics) {
                    this.triggerHapticFeedback(controller, 0.3, 30);
                }
            }
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
            if (distance < closestDistance && distance < (this.settings.ar.interactionRadius || 0.5)) {
                closestDistance = distance;
                closestNode = node;
            }
        }

        return closestNode?.id || null;
    }

    public dispose(): void {
        this.controllers = [];
        this.hands = [];
        this.lastInteractorPosition.set(0, 0, 0);
        this.selectedNodeId = null;
        XRInteraction.instance = null;
    }
}
