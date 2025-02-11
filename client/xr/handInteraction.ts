import { Vector3, Mesh } from 'three';
import { XRHandWithHaptics } from '../types/xr';
import { WebSocketService } from '../websocket/websocketService';
import { EnhancedNodeManager } from '../rendering/EnhancedNodeManager';
import { createLogger } from '../core/logger';
import { Node } from '../core/types';

const _logger = createLogger('HandInteraction');

export class HandInteractionManager {
    private static instance: HandInteractionManager;
    private lastPinchState: boolean = false;
    private websocketService: WebSocketService;
    private nodeManager?: EnhancedNodeManager;

    private constructor() {
        this.websocketService = WebSocketService.getInstance();
        // Note: nodeManager will be set via setNodeManager
    }

    public static getInstance(): HandInteractionManager {
        if (!HandInteractionManager.instance) {
            HandInteractionManager.instance = new HandInteractionManager();
        }
        return HandInteractionManager.instance;
    }

    public setNodeManager(nodeManager: EnhancedNodeManager): void {
        this.nodeManager = nodeManager;
    }

    public processHandInput(hand: XRHandWithHaptics): void {
        if (!this.nodeManager) return;

        const thumbTip = hand.hand.joints['thumb-tip'];
        const indexTip = hand.hand.joints['index-finger-tip'];

        if (!thumbTip || !indexTip) return;

        const distance = thumbTip.position.distanceTo(indexTip.position);
        const pinchStrength = Math.max(0, 1 - distance / 0.05); // 5cm max distance
        hand.pinchStrength = pinchStrength;

        // Detect pinch gesture
        const isPinching = pinchStrength > 0.9; // 90% threshold for pinch
        if (isPinching !== this.lastPinchState) {
            this.lastPinchState = isPinching;
            if (isPinching) {
                this.handlePinchGesture(indexTip.position);
            }
        }
    }

    private handlePinchGesture(position: Vector3): void {
        if (!this.nodeManager) return;

        // Find closest node to index finger tip
        const nodes = Array.from(this.nodeManager.getNodes().values());
        let closestNodeMesh: Mesh | null = null;
        let closestDistance = Infinity;

        for (const nodeMesh of nodes) {
            const nodePos = nodeMesh.position;
            const distance = nodePos.distanceTo(position);
            if (distance < closestDistance && distance < 0.1) { // 10cm threshold
                closestNodeMesh = nodeMesh;
                closestDistance = distance;
            }
        }

        const closestNode = closestNodeMesh?.userData as Node | undefined;
        if (!closestNode) return;

        if (closestNode && closestNode.id) {
            _logger.debug(`Pinch gesture detected on node ${closestNode.id}`);
            
            // Send node position update through websocket
            this.websocketService.sendNodeUpdates([{
                id: closestNode.id,
                position: {
                    x: position.x,
                    y: position.y,
                    z: position.z
                },
                velocity: {
                    x: 0,
                    y: 0,
                    z: 0
                }
            }]);

            // Also update local node position
            this.nodeManager.updateNodePositions([{ id: closestNode.id, data: { position: [position.x, position.y, position.z], velocity: [0, 0, 0] } }]);
        }
    }

    public dispose(): void {
        this.lastPinchState = false;
    }
}