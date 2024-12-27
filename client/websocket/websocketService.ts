import { createLogger } from '../core/logger';
import { buildWsUrl, buildApiUrl } from '../core/api';
import { API_ENDPOINTS } from '../core/constants';
import { convertObjectKeysToCamelCase } from '../core/utils';

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
            const settings = await this.loadWebSocketSettings();
            this.url = settings.url || buildWsUrl();
            
            if (!this.url) {
                throw new Error('No WebSocket URL available');
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
            }

            // Send initial protocol messages
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
            }
            
            this.handleReconnect();
        };

        this.ws.onmessage = async (event: MessageEvent): Promise<void> => {
            try {
                if (this.connectionState !== ConnectionState.CONNECTED || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                    logger.warn('WebSocket not connected, ignoring message');
                    return;
                }

                // Handle text messages
                if (typeof event.data === 'string') {
                    try {
                        const message = JSON.parse(event.data);
                        if (message.error) {
                            logger.error('[Server Error]', message.error);
                            return;
                        }
                        // Handle settings update
                        if (message.type === 'settings_update') {
                            this.handleSettingsUpdate(message);
                            return;
                        }
                    } catch (e) {
                        logger.warn('Received non-JSON text message:', event.data);
                        return;
                    }
                }

                // Handle binary position/velocity updates
                if (event.data instanceof ArrayBuffer && this.binaryMessageCallback) {
                    // Validate data length (24 bytes per node - 6 floats * 4 bytes)
                    if (event.data.byteLength % 24 !== 0) {
                        logger.error('Invalid binary message length:', event.data.byteLength);
                        return;
                    }

                    const float32Array = new Float32Array(event.data);
                    const nodeCount = float32Array.length / 6;
                    const nodes: NodeData[] = [];

                    for (let i = 0; i < nodeCount; i++) {
                        const baseIndex = i * 6;
                        
                        // Validate float values
                        const values = float32Array.slice(baseIndex, baseIndex + 6);
                        if (!values.every(v => Number.isFinite(v))) {
                            logger.error('Invalid float values in node data at index:', i);
                            continue;
                        }

                        nodes.push({
                            position: [
                                values[0],
                                values[1],
                                values[2]
                            ],
                            velocity: [
                                values[3],
                                values[4],
                                values[5]
                            ]
                        });
                    }

                    if (nodes.length > 0) {
                        await Promise.resolve().then(() => {
                            if (this.binaryMessageCallback) {
                                this.binaryMessageCallback(nodes);
                            }
                        });
                    }
                }
            } catch (error) {
                logger.error('Error processing WebSocket message:', error);
            }
        };
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
        const { category, setting, value } = message;
        
        // Use existing utilities for case conversion
        const convertedValue = convertObjectKeysToCamelCase({
            [category]: {
                [setting]: value
            }
        }) as Record<string, Record<string, unknown>>;
        
        // Extract the converted category, setting and value
        const entries = Object.entries(convertedValue);
        if (entries.length > 0) {
            const [camelCategory, settingsObj] = entries[0];
            const settingEntries = Object.entries(settingsObj);
            if (settingEntries.length > 0) {
                const [camelSetting, camelValue] = settingEntries[0];
                // Update settings store
                this.settingsStore.set(`${camelCategory}.${camelSetting}`, camelValue);
                
                // Notify settings update handler
                if (this.settingsUpdateHandler) {
                    this.settingsUpdateHandler({
                        [camelCategory]: {
                            [camelSetting]: camelValue
                        }
                    });
                }
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

    public getConnectionStatus(): ConnectionState {
        return this.connectionState;
    }

    private sendMessage(message: any): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(message));
            } catch (error) {
                logger.error('Error sending message:', error);
            }
        }
    }

    public sendNodeUpdates(updates: NodeUpdate[]): void {
        if (this.connectionState !== ConnectionState.CONNECTED || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            logger.warn('WebSocket not connected, cannot send node updates');
            return;
        }

        try {
            // Validate updates
            if (!Array.isArray(updates) || updates.length === 0) {
                logger.warn('Invalid node updates: empty or not an array');
                return;
            }

            // Validate each update
            const validUpdates = updates.filter(update => {
                if (!update.position || typeof update.position !== 'object') {
                    logger.warn('Invalid node update: missing or invalid position', update);
                    return false;
                }

                const { x, y, z } = update.position;
                if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
                    logger.warn('Invalid node update: non-finite position values', update);
                    return false;
                }

                return true;
            });

            if (validUpdates.length === 0) {
                logger.warn('No valid updates to send');
                return;
            }

            // Create binary message (24 bytes per node - 6 floats * 4 bytes)
            const float32Array = new Float32Array(validUpdates.length * 6);
            
            validUpdates.forEach((update, index) => {
                const baseIndex = index * 6;
                float32Array[baseIndex] = update.position.x;
                float32Array[baseIndex + 1] = update.position.y;
                float32Array[baseIndex + 2] = update.position.z;
                // Set velocity components to 0 as they're not provided in updates
                float32Array[baseIndex + 3] = 0;
                float32Array[baseIndex + 4] = 0;
                float32Array[baseIndex + 5] = 0;
            });

            this.ws.send(float32Array.buffer);
            logger.debug(`Sent ${validUpdates.length} node updates`);
        } catch (error) {
            logger.error('Error sending node updates:', error);
            if (error instanceof Error) {
                logger.error('Error details:', {
                    name: error.name,
                    message: error.message,
                    stack: error.stack
                });
            }
        }
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

    private async loadWebSocketSettings(): Promise<any> {
        try {
            // Try both endpoints
            const endpoints = [
                API_ENDPOINTS.WEBSOCKET_CONTROL,
                buildApiUrl('settings/websocket')
            ];

            let response = null;
            for (const endpoint of endpoints) {
                try {
                    response = await fetch(endpoint);
                    if (response.ok) break;
                } catch (e) {
                    continue;
                }
            }

            if (!response || !response.ok) {
                throw new Error('Failed to load WebSocket settings');
            }

            return await response.json();
        } catch (error) {
            logger.error('Failed to load WebSocket settings:', error);
            // Return defaults
            return {
                url: buildWsUrl()
            };
        }
    }
}
