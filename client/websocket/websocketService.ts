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

            // Send initial protocol messages
            logger.debug('Sending initial protocol messages');
            this.sendMessage({ type: 'requestInitialData' });
            this.sendMessage({ type: 'enableBinaryUpdates' });
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
            if (event.data instanceof Blob) {
                logger.debug('Received binary message');
                this.handleBinaryMessage(event.data);
            } else {
                try {
                    const message = JSON.parse(event.data);
                    logger.debug('Received JSON message:', message);
                    if (message.type === 'settings') {
                        this.handleSettingsUpdate(message);
                    }
                } catch (e) {
                    logger.error('Failed to parse WebSocket message:', e);
                }
            }
        };
    }

    private async handleBinaryMessage(blob: Blob): Promise<void> {
        try {
            const arrayBuffer = await blob.arrayBuffer();
            const dataView = new DataView(arrayBuffer);
            const nodeCount = dataView.getUint32(0, true); // true for little-endian
            const nodes: NodeData[] = [];
            
            let offset = 4; // Start after node count
            for (let i = 0; i < nodeCount; i++) {
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

                nodes.push({ position, velocity });
            }

            if (this.binaryMessageCallback) {
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

        const buffer = new ArrayBuffer(4 + updates.length * 24); // 4 bytes for count + 24 bytes per node (3 floats for position, 3 for velocity)
        const dataView = new DataView(buffer);
        
        dataView.setUint32(0, updates.length, true); // Set node count
        
        let offset = 4;
        updates.forEach(update => {
            // Position
            dataView.setFloat32(offset, update.position.x, true);
            dataView.setFloat32(offset + 4, update.position.y, true);
            dataView.setFloat32(offset + 8, update.position.z, true);
            offset += 12;

            // Velocity (set to 0 as it's not included in NodeUpdate)
            dataView.setFloat32(offset, 0, true);
            dataView.setFloat32(offset + 4, 0, true);
            dataView.setFloat32(offset + 8, 0, true);
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
