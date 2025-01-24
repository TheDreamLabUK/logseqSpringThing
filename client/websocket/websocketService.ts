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
                logger.debug('WebSocket readyState:', this.ws.readyState);
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
            if (event.data instanceof ArrayBuffer) {
                logger.debug('Received binary position update');
                this.handleBinaryMessage(event.data);
            } else if (typeof event.data === 'string') {
                try {
                    const message = JSON.parse(event.data);
                    logger.debug('Received JSON message:', message);
                    switch (message.type) {
                        case 'settings':
                            this.handleSettingsUpdate(message);
                            break;
                        case 'connection_established':
                        case 'updatesStarted':
                            logger.info(`WebSocket ${message.type}`);
                            break;
                        default:
                            logger.warn('Unknown message type:', message.type);
                    }
                } catch (e) {
                    logger.error('Failed to parse WebSocket message:', e);
                }
            } else {
                logger.warn('Received unknown message type:', typeof event.data);
            }
        };
    }

    // Message types matching server's binary protocol
    private readonly MessageType = {
        PositionUpdate: 0x01,
        VelocityUpdate: 0x02,
        FullStateUpdate: 0x03
    } as const;

    private handleBinaryMessage(buffer: ArrayBuffer): void {
        try {
            const dataView = new DataView(buffer);
            let offset = 0;

            // Read and validate message type
            const messageType = dataView.getUint32(offset, true);
            offset += 4;

            if (messageType !== this.MessageType.FullStateUpdate) {
                logger.warn('Unexpected binary message type:', messageType);
                return;
            }

            // Read node count
            const nodeCount = dataView.getUint32(offset, true);
            offset += 4;

            logger.debug(`Processing binary update with ${nodeCount} nodes`);
            const nodes: NodeData[] = [];
            
            // Read node data
            for (let i = 0; i < nodeCount; i++) {
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

                nodes.push({ id, position, velocity });
            }

            // Notify callback if registered
            if (this.binaryMessageCallback) {
                logger.debug('Notifying callback with', nodes.length, 'nodes');
                this.binaryMessageCallback(nodes);
            }
        } catch (e) {
            logger.error('Failed to process binary message:', e);
        }
    }

    private handleReconnect(): void {
        const wasConnected = this.connectionState === ConnectionState.CONNECTED;
        this.connectionState = ConnectionState.DISCONNECTED;
        this.binaryMessageCallback = null;
        
        if (this.reconnectTimeout !== null) {
            window.clearTimeout(this.reconnectTimeout);
        }
        
        if (this.reconnectAttempts < this._maxReconnectAttempts &&
            (wasConnected || this.reconnectAttempts === 0)) {
            
            this.reconnectAttempts++;
            const delay = this.getReconnectDelay();
            
            logger.info(
                `WebSocket connection closed, attempt ${this.reconnectAttempts}/${this._maxReconnectAttempts} in ${delay}ms`
            );
            
            this.connectionState = ConnectionState.RECONNECTING;
            
            this.reconnectTimeout = window.setTimeout(() => {
                this.reconnectTimeout = null;
                this.connect();
            }, delay);
        } else if (this.reconnectAttempts >= this._maxReconnectAttempts) {
            logger.warn('Maximum reconnection attempts reached, WebSocket disabled');
            this.connectionState = ConnectionState.FAILED;
            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(false);
            }
        } else {
            logger.info('WebSocket connection closed');
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
        dataView.setUint32(offset, this.MessageType.PositionUpdate, true);
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
