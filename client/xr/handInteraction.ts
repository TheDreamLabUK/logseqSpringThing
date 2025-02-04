import { Vector3 } from 'three';
import { XRHandWithHaptics } from '../types/xr';
import { WebSocketService } from '../websocket/websocketService';
import { NodeManager } from '../rendering/nodes';
import { createLogger } from '../core/logger';
import { Node } from '../core/types';

const _logger = createLogger('HandInteraction');

export class HandInteractionManager {
    private static instance: HandInteractionManager;
    private lastPinchState: boolean = false;
    private websocketService: WebSocketService;
    private nodeManager: NodeManager;

    private constructor() {
        this.websocketService = WebSocketService.getInstance();
        this.nodeManager = NodeManager.getInstance();
    }

    public static getInstance(): HandInteractionManager {
        if (!HandInteractionManager.instance) {
            HandInteractionManager.instance = new HandInteractionManager();
        }
        return HandInteractionManager.instance;
    }

    public processHandInput(hand: XRHandWithHaptics): void {
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
        // Find closest node to index finger tip
        const nodes = this.nodeManager.getCurrentNodes();
        let closestNode: Node | null = null;
        let closestDistance = Infinity;

        for (const node of nodes) {
            const nodePos = this.nodeManager.getNodePosition(node.id);
            const distance = nodePos.distanceTo(position);
            if (distance < closestDistance && distance < 0.1) { // 10cm threshold
                closestNode = node;
                closestDistance = distance;
            }
        }

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
            this.nodeManager.updateNodePosition(closestNode.id, position);
        }
    }

    public dispose(): void {
        this.lastPinchState = false;
    }
}