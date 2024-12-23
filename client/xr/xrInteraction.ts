import * as THREE from 'three';
import { NodeManager } from '../rendering/nodes';
import { XRSessionManager } from './xrSessionManager';
import { Settings } from '../core/types';
import { SettingsManager } from '../state/settings';
import { XRHandWithHaptics } from './xrTypes';

interface HapticActuator {
    pulse: (intensity: number, duration: number) => Promise<boolean>;
}

export class XRInteraction {
    private static instance: XRInteraction | null = null;
    private xrManager: XRSessionManager;
    private nodeManager: NodeManager;
    private controllers: THREE.Group[] = [];
    private lastInteractorPosition = new THREE.Vector3();
    private hands: XRHandWithHaptics[] = [];
    private settings: Settings;
    private settingsManager: SettingsManager;

    private constructor(xrManager: XRSessionManager, nodeManager: NodeManager, settingsManager: SettingsManager) {
        this.xrManager = xrManager;
        this.nodeManager = nodeManager;
        this.settingsManager = settingsManager;
        this.settings = this.settingsManager.getCurrentSettings();
        
        this.setupXRControllers();
        this.setupHandTracking();
    }

    public static getInstance(xrManager: XRSessionManager, nodeManager: NodeManager, settingsManager: SettingsManager): XRInteraction {
        if (!XRInteraction.instance) {
            XRInteraction.instance = new XRInteraction(xrManager, nodeManager, settingsManager);
        }
        return XRInteraction.instance;
    }

    private setupXRControllers(): void {
        this.xrManager.onControllerAdded((controller: THREE.Group) => {
            this.controllers.push(controller);
            if (controller.userData.hapticActuator) {
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
        if (!this.settings.ar.enableHandTracking) return;

        // Hand tracking is handled by the XRSessionManager directly
        this.hands = [];
    }

    public update(): void {
        if (!this.settings.ar.enableHandTracking) return;

        // Update hand interactions
        this.hands.forEach(hand => {
            if (hand.pinchStrength > this.settings.ar.pinchThreshold) {
                this.handlePinchGesture(hand);
            }
        });

        // Update controller interactions
        this.controllers.forEach(controller => {
            this.handleControllerInteraction(controller);
        });
    }

    private handlePinchGesture(hand: XRHandWithHaptics): void {
        const indexTip = hand.hand.joints['index-finger-tip'];
        if (!indexTip) return;

        const position = new THREE.Vector3();
        position.setFromMatrixPosition(indexTip.matrixWorld);

        // Calculate movement delta
        const delta = position.clone().sub(this.lastInteractorPosition);
        
        // Update node position based on hand movement
        if (delta.length() > this.settings.ar.dragThreshold) {
            // Get all nodes and update their positions
            const nodes = this.nodeManager.getAllNodeMeshes();
            nodes.forEach(nodeMesh => {
                const currentPos = this.nodeManager.getNodePosition(nodeMesh.userData.nodeId);
                const newPos = currentPos.add(delta);
                this.nodeManager.updateNodePosition(nodeMesh.userData.nodeId, newPos);
            });

            if (this.settings.ar.enableHaptics) {
                this.triggerHapticFeedback(hand, this.settings.ar.hapticIntensity, 50);
            }
        }

        this.lastInteractorPosition.copy(position);
    }

    private handleControllerInteraction(controller: THREE.Group): void {
        const position = new THREE.Vector3();
        position.setFromMatrixPosition(controller.matrixWorld);

        // Calculate movement delta
        const delta = position.clone().sub(this.lastInteractorPosition);
        
        // Update node position based on controller movement
        if (delta.length() > this.settings.ar.dragThreshold) {
            // Get all nodes and update their positions
            const nodes = this.nodeManager.getAllNodeMeshes();
            nodes.forEach(nodeMesh => {
                const currentPos = this.nodeManager.getNodePosition(nodeMesh.userData.nodeId);
                const newPos = currentPos.add(delta);
                this.nodeManager.updateNodePosition(nodeMesh.userData.nodeId, newPos);
            });

            if (this.settings.ar.enableHaptics && controller.userData.hapticActuator) {
                this.triggerHapticFeedback(controller, this.settings.ar.hapticIntensity, 50);
            }
        }

        this.lastInteractorPosition.copy(position);
    }

    private triggerHapticFeedback(device: THREE.Group | XRHandWithHaptics, intensity: number, duration: number): void {
        if (!this.settings.ar.enableHaptics) return;

        if ('hapticActuators' in device) {
            const hapticActuators = device.hapticActuators as HapticActuator[];
            hapticActuators.forEach((actuator: HapticActuator) => {
                actuator.pulse(intensity, duration).catch((error: Error) => {
                    console.warn('Failed to trigger haptic feedback:', error);
                });
            });
        } else if (device.userData.hapticActuator) {
            device.userData.hapticActuator.pulse(intensity, duration).catch((error: Error) => {
                console.warn('Failed to trigger haptic feedback:', error);
            });
        }
    }

    public dispose(): void {
        this.controllers = [];
        this.hands = [];
        XRInteraction.instance = null;
    }
}
