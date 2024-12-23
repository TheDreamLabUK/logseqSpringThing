import { createLogger } from '../core/logger';

const logger = createLogger('WebSocketService');

// Simple interface matching server's binary format
interface NodeData {
    position: [number, number, number];
    velocity: [number, number, number];
}

type BinaryMessageCallback = (nodes: NodeData[]) => void;

export class WebSocketService {
    private static instance: WebSocketService | null = null;
    private ws: WebSocket | null = null;
    private binaryMessageCallback: BinaryMessageCallback | null = null;
    private reconnectTimeout: number | null = null;
    private isConnected: boolean = false;

    private constructor() {
        this.setupWebSocket();
    }

    private setupWebSocket(): void {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/wss`;

        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            logger.info('WebSocket connected');
            this.isConnected = true;
        };

        this.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer && this.binaryMessageCallback) {
                // Process binary position/velocity updates
                const float32Array = new Float32Array(event.data);
                const nodeCount = float32Array.length / 6; // 6 floats per node
                const nodes: NodeData[] = [];

                for (let i = 0; i < nodeCount; i++) {
                    const baseIndex = i * 6;
                    nodes.push({
                        position: [
                            float32Array[baseIndex],
                            float32Array[baseIndex + 1],
                            float32Array[baseIndex + 2]
                        ],
                        velocity: [
                            float32Array[baseIndex + 3],
                            float32Array[baseIndex + 4],
                            float32Array[baseIndex + 5]
                        ]
                    });
                }

                this.binaryMessageCallback(nodes);
            }
        };

        this.ws.onerror = (error) => {
            logger.error('WebSocket error:', error);
            this.isConnected = false;
        };

        this.ws.onclose = () => {
            logger.info('WebSocket connection closed');
            this.isConnected = false;
            
            // Clear any existing reconnect timeout
            if (this.reconnectTimeout !== null) {
                window.clearTimeout(this.reconnectTimeout);
            }
            
            // Attempt to reconnect after a delay
            this.reconnectTimeout = window.setTimeout(() => {
                this.reconnectTimeout = null;
                this.setupWebSocket();
            }, 5000);
        };
    }

    public static getInstance(): WebSocketService {
        if (!WebSocketService.instance) {
            WebSocketService.instance = new WebSocketService();
        }
        return WebSocketService.instance;
    }

    public onBinaryMessage(callback: BinaryMessageCallback): void {
        this.binaryMessageCallback = callback;
    }

    public getConnectionStatus(): boolean {
        return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
    }

    public sendNodeUpdates(updates: Array<{ id: string; position: { x: number; y: number; z: number } }>): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            logger.warn('WebSocket not connected, cannot send node updates');
            return;
        }

        // Convert updates to Float32Array for efficient binary transmission
        const float32Array = new Float32Array(updates.length * 3); // 3 floats per position
        updates.forEach((update, index) => {
            const baseIndex = index * 3;
            float32Array[baseIndex] = update.position.x;
            float32Array[baseIndex + 1] = update.position.y;
            float32Array[baseIndex + 2] = update.position.z;
        });

        this.ws.send(float32Array.buffer);
    }

    public dispose(): void {
        if (this.reconnectTimeout !== null) {
            window.clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.binaryMessageCallback = null;
        this.isConnected = false;
    }
}
