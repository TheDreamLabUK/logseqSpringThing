import * as THREE from 'three';
import { XRHandWithHaptics } from './xrTypes';
import { NodeManager } from '../rendering/nodes';
import { XRSessionManager } from './xrSessionManager';
import { VisualizationSettings } from '../core/types';

export class XRInteraction {
    private static instance: XRInteraction | null = null;
    private xrManager: XRSessionManager;
    private nodeManager: NodeManager;
    private controllers: THREE.Group[] = [];
    private lastInteractorPosition = new THREE.Vector3();
    private hands: XRHandWithHaptics[] = [];
    private settings: VisualizationSettings;

    private constructor(xrManager: XRSessionManager, nodeManager: NodeManager) {
        this.xrManager = xrManager;
        this.nodeManager = nodeManager;
        this.settings = {
            enableHandTracking: true,
            pinchThreshold: 0.015,
            dragThreshold: 0.04,
            enableHaptics: true,
            hapticIntensity: 0.7
        } as VisualizationSettings;
        
        this.setupXRControllers();
        this.setupHandTracking();
    }

    public static getInstance(xrManager: XRSessionManager, nodeManager: NodeManager): XRInteraction {
        if (!XRInteraction.instance) {
            XRInteraction.instance = new XRInteraction(xrManager, nodeManager);
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
        if (!this.settings.enableHandTracking) return;

        // Hand tracking is handled by the XRSessionManager directly
        this.hands = [];
    }

    public update(): void {
        if (!this.settings.enableHandTracking) return;

        // Update hand interactions
        this.hands.forEach(hand => {
            if (hand.pinchStrength > this.settings.pinchThreshold) {
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
        
        // Update node positions based on hand movement
        if (delta.length() > this.settings.dragThreshold) {
            this.nodeManager.updateNodePositions({
                x: delta.x,
                y: delta.y,
                z: delta.z
            });
            if (this.settings.enableHaptics) {
                this.triggerHapticFeedback(hand, this.settings.hapticIntensity, 50);
            }
        }

        this.lastInteractorPosition.copy(position);
    }

    private handleControllerInteraction(controller: THREE.Group): void {
        const position = new THREE.Vector3();
        position.setFromMatrixPosition(controller.matrixWorld);

        // Calculate movement delta
        const delta = position.clone().sub(this.lastInteractorPosition);
        
        // Update node positions based on controller movement
        if (delta.length() > this.settings.dragThreshold) {
            this.nodeManager.updateNodePositions({
                x: delta.x,
                y: delta.y,
                z: delta.z
            });
            if (this.settings.enableHaptics && controller.userData.hapticActuator) {
                this.triggerHapticFeedback(controller, this.settings.hapticIntensity, 50);
            }
        }

        this.lastInteractorPosition.copy(position);
    }

    private triggerHapticFeedback(device: THREE.Group | XRHandWithHaptics, intensity: number, duration: number): void {
        if (!this.settings.enableHaptics) return;

        if ('hapticActuators' in device) {
            device.hapticActuators.forEach(actuator => {
                actuator.pulse(intensity, duration);
            });
        } else if (device.userData.hapticActuator) {
            device.userData.hapticActuator.pulse(intensity, duration);
        }
    }

    public dispose(): void {
        this.controllers = [];
        this.hands = [];
        XRInteraction.instance = null;
    }
}
