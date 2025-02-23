import { createLogger } from '../core/logger';
import { buildWsUrl } from '../core/api';
import { debugState } from '../core/debugState';
import pako from 'pako';

const logger = createLogger('WebSocketService');

// Helper for conditional debug logging
function debugLog(message: string, ...args: any[]) {
    if (debugState.isWebsocketDebugEnabled()) {
        logger.debug(message, ...args);
    }
}

// Compression settings
const COMPRESSION_THRESHOLD = 1024; // Only compress messages larger than 1KB

enum ConnectionState {
    DISCONNECTED = 'disconnected',
    CONNECTING = 'connecting',
    CONNECTED = 'connected',
    RECONNECTING = 'reconnecting',
    FAILED = 'failed'
}

// Interface for node updates from user interaction
interface NodeUpdate {
    id: string;          // Node ID (converted to u32 for binary protocol)
    position: {          // Current position
        x: number;
        y: number;
        z: number;
    };
    velocity?: {         // Optional velocity (defaults to 0 if not provided)
        x: number;
        y: number;
        z: number;
    };
}

// Interface matching server's binary protocol format (28 bytes per node):
// - id: 4 bytes (u32)
// - position: 12 bytes (3 × f32)
// - velocity: 12 bytes (3 × f32)
interface BinaryNodeData {
    id: number;
    position: [number, number, number];
    velocity: [number, number, number];
}

type BinaryMessageCallback = (nodes: BinaryNodeData[]) => void;

export class WebSocketService {
    private static instance: WebSocketService | null = null;
    private ws: WebSocket | null = null;
    private binaryMessageCallback: BinaryMessageCallback | null = null;
    private reconnectTimeout: number | null = null;
    private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
    private reconnectAttempts: number = 0;
    private readonly _maxReconnectAttempts: number = 5;
    private readonly initialReconnectDelay: number = 5000; // 5 seconds
    private readonly maxReconnectDelay: number = 60000; // 60 seconds
    private url: string = '';
    private connectionStatusHandler: ((status: boolean) => void) | null = null;
    private readonly MAX_POSITION = 1000.0;
    private readonly MAX_VELOCITY = 10.0;

    private validateAndClampValue(value: number, max: number): number {
        if (isNaN(value) || !isFinite(value)) {
            return 0;
        }
        return Math.max(Math.min(value, max), -max);
    }

    private validateAndClampVector3(vec: [number, number, number], max: number): [number, number, number] {
        return [
            this.validateAndClampValue(vec[0], max),
            this.validateAndClampValue(vec[1], max),
            this.validateAndClampValue(vec[2], max)
        ];
    }

    private constructor() {
        // Don't automatically connect - wait for explicit connect() call
    }

    public connect(): Promise<void> {
        if (this.connectionState !== ConnectionState.DISCONNECTED) {
            logger.warn('WebSocket already connected or connecting');
            // If already connecting, return a promise that resolves when connected
            if (this.connectionState === ConnectionState.CONNECTING) {
                return new Promise((resolve) => {
                    const checkConnection = () => {
                        if (this.connectionState === ConnectionState.CONNECTED) {
                            resolve();
                        } else {
                            setTimeout(checkConnection, 100);
                        }
                    };
                    checkConnection();
                });
            }
            return Promise.resolve();
        }
        return this.initializeWebSocket();
    }

    private async initializeWebSocket(): Promise<void> {
        if (this.connectionState !== ConnectionState.DISCONNECTED) {
            return;
        }

        try {
            this.url = buildWsUrl();
            
            if (!this.url) {
                throw new Error('No WebSocket URL available');
            }

            this.connectionState = ConnectionState.CONNECTING;
            return new Promise((resolve, reject) => {
                this.ws = new WebSocket(this.url);
                this.setupWebSocketHandlers();
                
                // Add one-time open handler to resolve the promise
                this.ws!.addEventListener('open', () => resolve(), { once: true });
                // Add one-time error handler to reject the promise
                this.ws!.addEventListener('error', (e) => reject(e), { once: true });
            });
        } catch (error) {
            logger.error('Failed to initialize WebSocket:', error);
            this.handleReconnect();
            return Promise.reject(error);
        }
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

    private setupWebSocketHandlers(): void {
        if (!this.ws) return;
        
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = (): void => {
            logger.info(`WebSocket connected successfully to ${this.url}`);
            this.connectionState = ConnectionState.CONNECTED;
            this.reconnectAttempts = 0;

            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(true);
                debugLog('Connection status handler notified: connected');
            }

            // Send request for position updates after connection
            debugLog('Requesting position updates');
            this.sendMessage({ type: 'requestInitialData' });
        };

        this.ws.onerror = (event: Event): void => {
            logger.error('WebSocket error:', event);
            if (this.ws?.readyState === WebSocket.CLOSED) {
                this.handleReconnect();
            }
        };

        this.ws.onclose = (event: CloseEvent): void => {
            logger.warn(`WebSocket closed with code ${event.code}: ${event.reason}`);
            
            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(false);
            }
            
            this.handleReconnect();
        };

        this.ws.onmessage = (event: MessageEvent) => {
            try {
                if (event.data instanceof ArrayBuffer) {
                    if (debugState.isWebsocketDebugEnabled()) {
                        debugLog('Received binary position update');
                    }
                    this.handleBinaryMessage(event.data);
                } else if (typeof event.data === 'string') {
                    try {
                        const message = JSON.parse(event.data);
                        if (message.type === 'connection_established' || message.type === 'updatesStarted') {
                            logger.info(`WebSocket ${message.type}`);
                        } else {
                            logger.warn('Unknown message type:', {
                                type: message.type,
                                message
                            });
                        }
                    } catch (error) {
                        logger.error('Failed to parse WebSocket message:', error);
                    }
                }
            } catch (error) {
                logger.error('Critical error in message handler:', error);
            }
        };
    }

    private tryDecompress(buffer: ArrayBuffer): ArrayBuffer {
        try {
            const decompressed = pako.inflate(new Uint8Array(buffer));
            if (decompressed.length < 8 || decompressed.length % 4 !== 0) {
                return buffer;
            }
            return decompressed.buffer;
        } catch (error) {
            return buffer;
        }
    }

    private compressIfNeeded(buffer: ArrayBuffer): ArrayBuffer {
        if (buffer.byteLength > COMPRESSION_THRESHOLD) {
            try {
                const compressed = pako.deflate(new Uint8Array(buffer));
                return compressed.buffer;
            } catch (error) {
                logger.warn('Compression failed, using original data:', error);
                return buffer;
            }
        }
        return buffer;
    }

    private handleBinaryMessage(buffer: ArrayBuffer): void {
        try {
            if (debugState.isWebsocketDebugEnabled()) {
                debugLog('Processing binary message:', { size: buffer.byteLength });
            }

            const decompressedBuffer = this.tryDecompress(buffer);
            if (debugState.isWebsocketDebugEnabled()) {
                debugLog('After decompression:', { size: decompressedBuffer.byteLength });
            }
            
            // Each node update is 28 bytes (4 for id, 12 for position, 12 for velocity)
            if (!decompressedBuffer || decompressedBuffer.byteLength % 28 !== 0) {
                throw new Error(`Invalid buffer size: ${decompressedBuffer?.byteLength ?? 0} bytes (not a multiple of 28)`);
            }

            const dataView = new DataView(decompressedBuffer);
            const nodeCount = decompressedBuffer.byteLength / 28;
            if (debugState.isWebsocketDebugEnabled()) {
                debugLog('Node count:', { count: nodeCount });
            }
            let offset = 0;
            let invalidValuesFound = false;
            const nodes: BinaryNodeData[] = [];
            
            for (let i = 0; i < nodeCount; i++) {
                const id = dataView.getUint32(offset, true);
                offset += 4;

                const position: [number, number, number] = [
                    dataView.getFloat32(offset, true),
                    dataView.getFloat32(offset + 4, true),
                    dataView.getFloat32(offset + 8, true)
                ];
                offset += 12;

                const velocity: [number, number, number] = [
                    dataView.getFloat32(offset, true),
                    dataView.getFloat32(offset + 4, true),
                    dataView.getFloat32(offset + 8, true)
                ];
                offset += 12;
                
                // Validate and clamp position and velocity
                const sanitizedPosition = this.validateAndClampVector3(position, this.MAX_POSITION);
                const sanitizedVelocity = this.validateAndClampVector3(velocity, this.MAX_VELOCITY);
                
                // Check if values were invalid
                if (sanitizedPosition.some((v, i) => v !== position[i]) ||
                    sanitizedVelocity.some((v, i) => v !== velocity[i])) {
                    invalidValuesFound = true;
                    logger.warn('Invalid values detected in binary message:', {
                        nodeId: id,
                        originalPosition: position,
                        sanitizedPosition,
                        originalVelocity: velocity,
                        sanitizedVelocity
                    });
                }

                nodes.push({ id, position: sanitizedPosition, velocity: sanitizedVelocity });
            }

            if (invalidValuesFound) {
                logger.warn('Some nodes had invalid position/velocity values that were clamped');
            }

            if (nodes.length > 0 && this.binaryMessageCallback) {
                if (debugState.isWebsocketDebugEnabled()) {
                    debugLog('Calling binary message callback with nodes:', { count: nodes.length });
                }
                this.binaryMessageCallback(nodes);
            } else {
                if (debugState.isWebsocketDebugEnabled()) {
                    debugLog('No nodes to process or no callback registered', {
                        nodesLength: nodes.length,
                        hasCallback: !!this.binaryMessageCallback
                    });
                }
            }
        } catch (error) {
            logger.error('Failed to process binary message:', error);
        }
    }

    private handleReconnect(): void {
        const wasConnected = this.connectionState === ConnectionState.CONNECTED;
        
        this.connectionState = ConnectionState.DISCONNECTED;
        this.binaryMessageCallback = null;
        
        if (this.reconnectTimeout !== null) {
            window.clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
        
        if (this.reconnectAttempts < this._maxReconnectAttempts &&
            (wasConnected || this.reconnectAttempts === 0)) {
            
            this.reconnectAttempts++;
            const delay = this.getReconnectDelay();
            
            this.connectionState = ConnectionState.RECONNECTING;
            
            this.reconnectTimeout = window.setTimeout(async () => {
                this.reconnectTimeout = null;
                try {
                    await this.connect();
                } catch (error) {
                    logger.error('Reconnection attempt failed:', error);
                }
            }, delay);
        } else {
            this.handleReconnectFailure();
        }
    }

    private handleReconnectFailure(): void {
        this.connectionState = ConnectionState.FAILED;
        if (this.connectionStatusHandler) {
            this.connectionStatusHandler(false);
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

    public getConnectionStatus(): ConnectionState {
        return this.connectionState;
    }

    public sendMessage(message: any): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(message));
            } catch (error) {
                logger.error('Error sending message:', error);
            }
        }
    }

    public sendNodeUpdates(updates: NodeUpdate[]): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            logger.warn('WebSocket not connected, cannot send node updates');
            return;
        }

        // Limit to 2 nodes per update as per server requirements
        if (updates.length > 2) {
            logger.warn('Too many nodes in update, limiting to first 2');
            updates = updates.slice(0, 2);
        }

        const buffer = new ArrayBuffer(updates.length * 28);
        const dataView = new DataView(buffer);
        let offset = 0;

        updates.forEach(update => {
            const id = parseInt(update.id, 10);
            if (isNaN(id)) {
                logger.warn('Invalid node ID:', update.id);
                return;
            }
            dataView.setUint32(offset, id, true);
            offset += 4;

            dataView.setFloat32(offset, update.position.x, true);
            dataView.setFloat32(offset + 4, update.position.y, true);
            dataView.setFloat32(offset + 8, update.position.z, true);
            offset += 12;

            dataView.setFloat32(offset, update.velocity?.x ?? 0, true);
            dataView.setFloat32(offset + 4, update.velocity?.y ?? 0, true);
            dataView.setFloat32(offset + 8, update.velocity?.z ?? 0, true);
            offset += 12;
        });

        const finalBuffer = this.compressIfNeeded(buffer);
        this.ws.send(finalBuffer);
    }

    public onConnectionStatusChange(handler: (status: boolean) => void): void {
        this.connectionStatusHandler = handler;
        if (this.connectionState === ConnectionState.CONNECTED && handler) {
            handler(true);
        }
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
        this.connectionStatusHandler = null;
        this.connectionState = ConnectionState.DISCONNECTED;
        WebSocketService.instance = null;
    }

    public close(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}