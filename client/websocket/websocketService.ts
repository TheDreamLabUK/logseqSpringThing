import { createLogger } from '../core/logger';
import { buildWsUrl } from '../core/api';

const logger = createLogger('WebSocketService');

enum ConnectionState {
    DISCONNECTED = 'disconnected',
    CONNECTING = 'connecting',
    CONNECTED = 'connected',
    RECONNECTING = 'reconnecting',
    FAILED = 'failed'
}

// Interface matching server's binary protocol format (28 bytes per node):
// - id: 4 bytes (u32)
// - position: 12 bytes (3 × f32)
// - velocity: 12 bytes (3 × f32)
interface NodeData {
    id: number;
    position: [number, number, number];
    velocity: [number, number, number];
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

interface SettingsUpdateMessage {
    category: string;
    setting: string;
    value: any;
}

type BinaryMessageCallback = (nodes: NodeData[]) => void;

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
    private settingsStore: Map<string, any> = new Map();
    private connectionStatusHandler: ((status: boolean) => void) | null = null;
    private settingsUpdateHandler: ((settings: any) => void) | null = null;

    private constructor() {
        // Don't automatically connect - wait for explicit connect() call
    }

    public connect(): void {
        if (this.connectionState !== ConnectionState.DISCONNECTED) {
            logger.warn('WebSocket already connected or connecting');
            return;
        }
        this.initializeWebSocket();
    }

    private async initializeWebSocket(): Promise<void> {
        if (this.connectionState !== ConnectionState.DISCONNECTED) {
            return;
        }

        try {
            // Always use buildWsUrl() to ensure proper protocol and path
            this.url = buildWsUrl();
            
            if (!this.url) {
                throw new Error('No WebSocket URL available');
            }

            // Ensure URL uses wss:// protocol when on HTTPS
            if (window.location.protocol === 'https:' && !this.url.startsWith('wss://')) {
                this.url = this.url.replace('ws://', 'wss://');
            }

            this.connectionState = ConnectionState.CONNECTING;
            this.ws = new WebSocket(this.url);
            this.setupWebSocketHandlers();
        } catch (error) {
            logger.error('Failed to initialize WebSocket:', error);
            this.handleReconnect();
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

            // Notify connection status change
            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(true);
                logger.debug('Connection status handler notified: connected');
            }

            // Send request for position updates after connection
            logger.debug('Requesting position updates');
            this.sendMessage({ type: 'requestInitialData' });
        };

        this.ws.onerror = (event: Event): void => {
            logger.error('WebSocket error:', event);
            if (this.ws) {
                logger.debug('Connection details:', {
                    readyState: this.ws.readyState,
                    url: this.url,
                    connectionState: this.connectionState,
                    reconnectAttempts: this.reconnectAttempts
                });
            }
            if (this.ws?.readyState === WebSocket.CLOSED) {
                this.handleReconnect();
            }
        };

        this.ws.onclose = (event: CloseEvent): void => {
            logger.warn(`WebSocket closed with code ${event.code}: ${event.reason}`);
            
            // Notify connection status change
            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(false);
                logger.debug('Connection status handler notified: disconnected');
            }
            
            this.handleReconnect();
        };

        this.ws.onmessage = (event: MessageEvent) => {
            try {
                if (event.data instanceof ArrayBuffer) {
                    logger.debug('Received binary position update');
                    try {
                        this.handleBinaryMessage(event.data);
                    } catch (error) {
                        logger.error('Failed to process binary message:', {
                            error,
                            dataSize: event.data.byteLength,
                            connectionState: this.connectionState
                        });
                    }
                } else if (typeof event.data === 'string') {
                    try {
                        const message = JSON.parse(event.data);
                        logger.debug('Received JSON message:', message);
                        
                        switch (message.type) {
                            case 'settings':
                                try {
                                    this.handleSettingsUpdate(message);
                                } catch (error) {
                                    logger.error('Failed to handle settings update:', {
                                        error,
                                        message,
                                        connectionState: this.connectionState
                                    });
                                }
                                break;
                            case 'connection_established':
                            case 'updatesStarted':
                                logger.info(`WebSocket ${message.type}`);
                                break;
                            default:
                                logger.warn('Unknown message type:', {
                                    type: message.type,
                                    message
                                });
                        }
                    } catch (error) {
                        logger.error('Failed to parse WebSocket message:', {
                            error,
                            data: event.data.slice(0, 200), // Log first 200 chars only
                            connectionState: this.connectionState
                        });
                    }
                } else {
                    logger.warn('Received unknown message type:', {
                        type: typeof event.data,
                        connectionState: this.connectionState
                    });
                }
            } catch (error) {
                logger.error('Critical error in message handler:', {
                    error,
                    connectionState: this.connectionState,
                    wsState: this.ws?.readyState
                });
            }
        };
    }

    // Message type matching server's binary protocol
    private readonly MessageType = {
        PositionVelocityUpdate: 0x01
    } as const;

    private handleBinaryMessage(buffer: ArrayBuffer): void {
        try {
            if (!buffer || buffer.byteLength < 8) {
                throw new Error(`Invalid buffer size: ${buffer?.byteLength ?? 0} bytes`);
            }

            const dataView = new DataView(buffer);
            let offset = 0;

            // Read and validate message type
            const messageType = dataView.getUint32(offset, true);
            offset += 4;

            if (messageType !== this.MessageType.PositionVelocityUpdate) {
                logger.warn('Unexpected binary message type:', {
                    received: messageType,
                    expected: this.MessageType.PositionVelocityUpdate,
                    bufferSize: buffer.byteLength
                });
                return;
            }

            // Read and validate node count
            const nodeCount = dataView.getUint32(offset, true);
            offset += 4;

            // Validate total message size
            const expectedSize = 8 + (nodeCount * 28); // 8 bytes header + 28 bytes per node
            if (buffer.byteLength !== expectedSize) {
                throw new Error(`Invalid buffer size: ${buffer.byteLength} bytes (expected ${expectedSize})`);
            }

            logger.debug('Processing binary update:', {
                nodeCount,
                messageType,
                bufferSize: buffer.byteLength
            });

            const nodes: NodeData[] = [];
            
            // Read node data
            for (let i = 0; i < nodeCount; i++) {
                try {
                    // Read node ID
                    const id = dataView.getUint32(offset, true);
                    offset += 4;

                    // Read position vector
                    const position: [number, number, number] = [
                        dataView.getFloat32(offset, true),
                        dataView.getFloat32(offset + 4, true),
                        dataView.getFloat32(offset + 8, true)
                    ];
                    offset += 12;

                    // Read velocity vector
                    const velocity: [number, number, number] = [
                        dataView.getFloat32(offset, true),
                        dataView.getFloat32(offset + 4, true),
                        dataView.getFloat32(offset + 8, true)
                    ];
                    offset += 12;

                    // Validate node data
                    if (position.some(isNaN) || velocity.some(isNaN)) {
                        throw new Error(`Invalid node data at index ${i}: NaN values detected`);
                    }

                    nodes.push({ id, position, velocity });
                } catch (nodeError) {
                    logger.error('Error processing node:', {
                        error: nodeError,
                        nodeIndex: i,
                        offset,
                        bufferSize: buffer.byteLength
                    });
                    // Continue processing other nodes
                }
            }

            if (nodes.length > 0) {
                // Notify callback if registered
                if (this.binaryMessageCallback) {
                    logger.debug('Notifying callback:', {
                        nodeCount: nodes.length,
                        firstNode: nodes[0],
                        lastNode: nodes[nodes.length - 1]
                    });
                    try {
                        this.binaryMessageCallback(nodes);
                    } catch (error) {
                        logger.error('Error in binary message callback:', {
                            error,
                            nodeCount: nodes.length,
                            connectionState: this.connectionState
                        });
                    }
                }
            } else {
                logger.warn('No valid nodes processed from binary message');
            }
        } catch (error) {
            logger.error('Failed to process binary message:', {
                error,
                bufferSize: buffer?.byteLength,
                connectionState: this.connectionState
            });
        }
    }

    private handleReconnect(): void {
        try {
            const wasConnected = this.connectionState === ConnectionState.CONNECTED;
            const previousState = this.connectionState;
            
            logger.debug('Handling reconnect:', {
                wasConnected,
                previousState,
                attempts: this.reconnectAttempts,
                maxAttempts: this._maxReconnectAttempts,
                url: this.url
            });

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
                
                logger.info('Scheduling reconnection:', {
                    attempt: this.reconnectAttempts,
                    maxAttempts: this._maxReconnectAttempts,
                    delay,
                    url: this.url
                });
                
                this.connectionState = ConnectionState.RECONNECTING;
                
                this.reconnectTimeout = window.setTimeout(() => {
                    try {
                        this.reconnectTimeout = null;
                        this.connect();
                    } catch (error) {
                        logger.error('Failed to initiate reconnection:', {
                            error,
                            attempts: this.reconnectAttempts,
                            state: this.connectionState
                        });
                        this.handleReconnectFailure();
                    }
                }, delay);
            } else if (this.reconnectAttempts >= this._maxReconnectAttempts) {
                logger.warn('Maximum reconnection attempts reached:', {
                    attempts: this.reconnectAttempts,
                    maxAttempts: this._maxReconnectAttempts,
                    url: this.url
                });
                this.handleReconnectFailure();
            } else {
                logger.info('WebSocket connection closed without reconnection', {
                    wasConnected,
                    attempts: this.reconnectAttempts
                });
            }
        } catch (error) {
            logger.error('Critical error in reconnect handler:', {
                error,
                connectionState: this.connectionState,
                attempts: this.reconnectAttempts
            });
            this.handleReconnectFailure();
        }
    }

    private handleReconnectFailure(): void {
        this.connectionState = ConnectionState.FAILED;
        if (this.connectionStatusHandler) {
            try {
                this.connectionStatusHandler(false);
            } catch (error) {
                logger.error('Error in connection status handler during failure:', error);
            }
        }
    }

    private handleSettingsUpdate(message: SettingsUpdateMessage): void {
        try {
            const { category, setting, value } = message;
            const settingsKey = `${category}.${setting}`;
            
            // Update local settings store
            this.settingsStore.set(settingsKey, value);

            // Notify settings update handler
            if (this.settingsUpdateHandler) {
                const settings = this.getSettingsSnapshot();
                this.settingsUpdateHandler(settings);
            }

            logger.debug(`Updated setting ${settingsKey}:`, value);
        } catch (e) {
            logger.error('Failed to handle settings update:', e);
        }
    }

    private getSettingsSnapshot(): any {
        const settings: any = {};
        for (const [key, value] of this.settingsStore.entries()) {
            const [category, setting] = key.split('.');
            if (!settings[category]) {
                settings[category] = {};
            }
            settings[category][setting] = value;
        }
        return settings;
    }

    public updateSettings(category: string, setting: string, value: any): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            logger.warn('WebSocket not connected, cannot update settings');
            return;
        }

        const message = {
            type: 'settings_update',
            category,
            setting,
            value
        };

        this.sendMessage(message);
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

        // 8 bytes header (4 for type + 4 for count) + 28 bytes per node (4 for id + 24 for position/velocity)
        const buffer = new ArrayBuffer(8 + updates.length * 28);
        const dataView = new DataView(buffer);
        let offset = 0;

        // Write message type (PositionUpdate)
        dataView.setUint32(offset, this.MessageType.PositionVelocityUpdate, true);
        offset += 4;

        // Write node count
        dataView.setUint32(offset, updates.length, true);
        offset += 4;

        updates.forEach(update => {
            // Write node ID
            const id = parseInt(update.id, 10);
            if (isNaN(id)) {
                logger.warn('Invalid node ID:', update.id);
                return;
            }
            dataView.setUint32(offset, id, true);
            offset += 4;

            // Write position
            dataView.setFloat32(offset, update.position.x, true);
            dataView.setFloat32(offset + 4, update.position.y, true);
            dataView.setFloat32(offset + 8, update.position.z, true);
            offset += 12;

            // Write velocity (use provided velocity or default to 0)
            dataView.setFloat32(offset, update.velocity?.x ?? 0, true);
            dataView.setFloat32(offset + 4, update.velocity?.y ?? 0, true);
            dataView.setFloat32(offset + 8, update.velocity?.z ?? 0, true);
            offset += 12;
        });

        this.ws.send(buffer);
    }

    public onConnectionStatusChange(handler: (status: boolean) => void): void {
        this.connectionStatusHandler = handler;
        // Immediately call handler with current status if connected
        if (this.connectionState === ConnectionState.CONNECTED && handler) {
            handler(true);
        }
    }

    public onSettingsUpdate(handler: (settings: any) => void): void {
        this.settingsUpdateHandler = handler;
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
        this.settingsUpdateHandler = null;
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
