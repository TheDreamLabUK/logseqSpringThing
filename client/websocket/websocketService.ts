import { createLogger } from '../core/logger';

const logger = createLogger('WebSocketService');

// Simple interface matching server's binary format
interface NodeData {
    position: [number, number, number];
    velocity: [number, number, number];
}

interface NodeUpdate {
    id: string;
    position: {
        x: number;
        y: number;
        z: number;
    };
}

type BinaryMessageCallback = (nodes: NodeData[]) => void;

export class WebSocketService {
    private static instance: WebSocketService | null = null;
    private ws: WebSocket | null = null;
    private binaryMessageCallback: BinaryMessageCallback | null = null;
    private reconnectTimeout: number | null = null;
    private isConnected: boolean = false;
    private reconnectAttempts: number = 0;
    private readonly maxReconnectAttempts: number = 5;
    private readonly initialReconnectDelay: number = 5000; // 5 seconds
    private readonly maxReconnectDelay: number = 60000; // 60 seconds

    private constructor() {
        this.setupWebSocket();
    }

    private getReconnectDelay(): number {
        // Exponential backoff with max delay
        const delay = Math.min(
            this.initialReconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );
        // Add some jitter
        return delay + (Math.random() * 1000);
    }

    private setupWebSocket(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            logger.warn('Maximum reconnection attempts reached, WebSocket disabled');
            return;
        }

        try {
            // Use relative path to avoid hardcoded domains
            const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/wss`;

            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = (): void => {
                logger.info('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0; // Reset counter on successful connection
            };

            this.ws.onmessage = async (event: MessageEvent): Promise<void> => {
                try {
                    if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                        logger.warn('WebSocket not connected, ignoring message');
                        return;
                    }

                    // Handle binary position/velocity updates
                    if (event.data instanceof ArrayBuffer && this.binaryMessageCallback) {
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

                        await Promise.resolve().then(() => {
                            if (this.binaryMessageCallback) {
                                this.binaryMessageCallback(nodes);
                            }
                        });
                    }
                } catch (error) {
                    logger.error('Error processing WebSocket message:', error);
                }
            };

            this.ws.onerror = (event: Event): void => {
                logger.error('WebSocket error:', event);
                this.isConnected = false;
            };

            this.ws.onclose = (event: CloseEvent): void => {
                const wasConnected = this.isConnected;
                this.isConnected = false;
                this.binaryMessageCallback = null;
                
                // Clear any existing reconnect timeout
                if (this.reconnectTimeout !== null) {
                    window.clearTimeout(this.reconnectTimeout);
                }
                
                // Only attempt to reconnect if:
                // 1. The close wasn't intentional (code !== 1000)
                // 2. We haven't exceeded max attempts
                // 3. We were previously connected (to avoid retry spam on initial failure)
                if (event.code !== 1000 && 
                    this.reconnectAttempts < this.maxReconnectAttempts &&
                    (wasConnected || this.reconnectAttempts === 0)) {
                    
                    this.reconnectAttempts++;
                    const delay = this.getReconnectDelay();
                    
                    logger.info(
                        `WebSocket connection closed (${event.code}), attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
                    );
                    
                    this.reconnectTimeout = window.setTimeout(() => {
                        this.reconnectTimeout = null;
                        this.setupWebSocket();
                    }, delay);
                } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    logger.warn('Maximum reconnection attempts reached, WebSocket disabled');
                } else {
                    logger.info(`WebSocket connection closed: ${event.code} ${event.reason}`);
                }
            };
        } catch (error) {
            logger.error('Failed to setup WebSocket:', error);
            this.isConnected = false;
            
            // Attempt to reconnect if we haven't exceeded max attempts
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = this.getReconnectDelay();
                
                logger.info(
                    `WebSocket setup failed, attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
                );
                
                this.reconnectTimeout = window.setTimeout(() => {
                    this.reconnectTimeout = null;
                    this.setupWebSocket();
                }, delay);
            }
        }
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

    public sendNodeUpdates(updates: NodeUpdate[]): void {
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
        WebSocketService.instance = null;
    }
}
