import * as THREE from 'three';
import { XRHandWithHaptics } from './xrTypes';
import { NodeManager } from '../rendering/nodes';
import { XRSessionManager } from './xrSessionManager';
import { Settings } from '../core/types';

export class XRInteraction {
    private static instance: XRInteraction | null = null;
    private xrManager: XRSessionManager;
    private nodeManager: NodeManager;
    private controllers: THREE.Group[] = [];
    private lastInteractorPosition = new THREE.Vector3();
    private hands: XRHandWithHaptics[] = [];
    private settings: Settings;

    private constructor(xrManager: XRSessionManager, nodeManager: NodeManager) {
        this.xrManager = xrManager;
        this.nodeManager = nodeManager;
        
        this.settings = {
            animations: { 
                enableMotionBlur: false, 
                enableNodeAnimations: false, 
                motionBlurStrength: 0.4, 
                selectionWaveEnabled: false, 
                pulseEnabled: false, 
                rippleEnabled: false, 
                edgeAnimationEnabled: false, 
                flowParticlesEnabled: false 
            },
            ar: {
                dragThreshold: 0.04,
                enableHandTracking: true,
                enableHaptics: true,
                enableLightEstimation: true,
                enablePassthroughPortal: false,
                enablePlaneDetection: true,
                enableSceneUnderstanding: true,
                gestureSsmoothing: 0.9,
                handMeshColor: '#FFD700',
                handMeshEnabled: true,
                handMeshOpacity: 0.3,
                handPointSize: 0.01,
                handRayColor: '#FFD700',
                handRayEnabled: true,
                handRayWidth: 0.002,
                hapticIntensity: 0.7,
                passthroughBrightness: 1,
                passthroughContrast: 1,
                passthroughOpacity: 1,
                pinchThreshold: 0.015,
                planeColor: '#4A90E2',
                planeOpacity: 0.3,
                portalEdgeColor: '#FFD700',
                portalEdgeWidth: 0.02,
                portalSize: 1,
                roomScale: true,
                rotationThreshold: 0.08,
                showPlaneOverlay: true,
                snapToFloor: true
            },
            audio: { 
                enableAmbientSounds: false, 
                enableInteractionSounds: false, 
                enableSpatialAudio: false 
            },
            bloom: { 
                edgeBloomStrength: 0.3, 
                enabled: false, 
                environmentBloomStrength: 0.5, 
                nodeBloomStrength: 0.2, 
                radius: 0.5, 
                strength: 1.8 
            },
            clientDebug: { 
                enableDataDebug: false, 
                enableWebsocketDebug: false, 
                enabled: false, 
                logBinaryHeaders: false, 
                logFullJson: false 
            },
            edges: {
                arrowSize: 0.15, 
                baseWidth: 2, 
                color: '#917f18', 
                enableArrows: false, 
                opacity: 1, 
                widthRange: [1, 4] 
            },
            labels: {
                desktopFontSize: 12, 
                enableLabels: true, 
                textColor: '#FFFFFF' 
            },
            network: {
                bindAddress: '0.0.0.0',
                domain: 'localhost',
                port: 3001
            },
            nodes: { 
                baseColor: '#4A90E2', 
                baseSize: 1, 
                clearcoat: 0.5, 
                enableHoverEffect: true, 
                enableInstancing: true, 
                highlightColor: '#FFD700', 
                highlightDuration: 500, 
                hoverScale: 1.2, 
                materialType: 'standard', 
                metalness: 0.5, 
                opacity: 1, 
                roughness: 0.5, 
                sizeByConnections: false, 
                sizeRange: [0.5, 2] 
            },
            physics: { 
                attractionStrength: 0.1, 
                boundsSize: 100, 
                collisionRadius: 1, 
                damping: 0.5, 
                enableBounds: true, 
                enabled: true, 
                iterations: 1, 
                maxVelocity: 10, 
                repulsionStrength: 0.2, 
                springStrength: 0.1 
            },
            rendering: { 
                ambientLightIntensity: 0.5, 
                backgroundColor: '#000000', 
                directionalLightIntensity: 1, 
                enableAmbientOcclusion: true, 
                enableAntialiasing: true, 
                enableShadows: true, 
                environmentIntensity: 1 
            },
            websocket: {
                heartbeatInterval: 30,
                heartbeatTimeout: 60,
                maxMessageSize: 5242880
            }
        };
        
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
