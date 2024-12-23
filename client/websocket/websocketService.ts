import { createLogger } from '../core/logger';

const logger = createLogger('WebSocketService');

interface NodeUpdate {
    id: string;
    position: {
        x: number;
        y: number;
        z: number;
    };
}

type BinaryMessageCallback = (positions: Float32Array) => void;

export class WebSocketService {
    private static instance: WebSocketService | null = null;
    private ws: WebSocket | null = null;
    private binaryMessageCallback: BinaryMessageCallback | null = null;

    private constructor() {
        this.setupWebSocket();
    }

    private setupWebSocket(): void {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/wss`;

        this.ws = new WebSocket(wsUrl);
        
        this.ws.binaryType = 'arraybuffer';

        this.ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer && this.binaryMessageCallback) {
                const positions = new Float32Array(event.data);
                this.binaryMessageCallback(positions);
            }
        };

        this.ws.onerror = (error) => {
            logger.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            logger.info('WebSocket connection closed');
            // Attempt to reconnect after a delay
            setTimeout(() => this.setupWebSocket(), 5000);
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

    public sendNodeUpdates(updates: NodeUpdate[]): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            logger.warn('WebSocket not connected');
            return;
        }

        try {
            this.ws.send(JSON.stringify({
                type: 'nodeUpdates',
                data: updates
            }));
        } catch (error) {
            logger.error('Failed to send node updates:', error);
        }
    }

    public dispose(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.binaryMessageCallback = null;
    }
}
