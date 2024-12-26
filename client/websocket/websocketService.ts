import { createLogger } from '../core/logger';
import { buildWsUrl } from '../core/api';
import { 
    WS_HEARTBEAT_INTERVAL,
    WS_HEARTBEAT_TIMEOUT,
    WS_RECONNECT_INTERVAL,
    WS_MESSAGE_QUEUE_SIZE,
    FLOATS_PER_NODE,
    NODE_POSITION_SIZE
} from '../core/constants';
import { convertObjectKeysToCamelCase } from '../core/utils';

const logger = createLogger('WebSocketService');

interface WebSocketError {
    error: string;
}

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
    private connectionMonitorHandle: number | null = null;
    private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
    private reconnectAttempts: number = 0;
    private lastPongTime: number = 0;
    private readonly maxReconnectAttempts: number = 5;
    private readonly initialReconnectDelay: number = 5000; // 5 seconds
    private readonly maxReconnectDelay: number = 60000; // 60 seconds
    private readonly heartbeatTimeout: number = WS_HEARTBEAT_TIMEOUT;
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

    private startConnectionMonitor(): void {
        // Clear any existing monitor
        if (this.connectionMonitorHandle !== null) {
            window.clearInterval(this.connectionMonitorHandle);
        }

        // Monitor connection health every heartbeat interval
        this.connectionMonitorHandle = window.setInterval(() => {
            if (this.connectionState !== ConnectionState.CONNECTED || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                this.stopConnectionMonitor();
                return;
            }

            const timeSinceLastPong = Date.now() - this.lastPongTime;
            if (timeSinceLastPong > this.heartbeatTimeout) {
                logger.warn('WebSocket heartbeat timeout, closing connection');
                this.ws.close();
                this.stopConnectionMonitor();
            }
        }, WS_HEARTBEAT_INTERVAL);
    }

    private stopConnectionMonitor(): void {
        if (this.connectionMonitorHandle !== null) {
            window.clearInterval(this.connectionMonitorHandle);
            this.connectionMonitorHandle = null;
        }
    }

    private setupWebSocket(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            logger.warn('Maximum reconnection attempts reached, WebSocket disabled');
            this.connectionState = ConnectionState.FAILED;
            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(false);
            }
            return;
        }

        try {
            const wsUrl = buildWsUrl();
            const isProduction = ['www.visionflow.info', 'visionflow.info'].includes(window.location.hostname);
            
            logger.info(`[${isProduction ? 'PROD' : 'DEV'}] Attempting WebSocket connection to ${wsUrl}`);

            this.connectionState = ConnectionState.CONNECTING;
            
            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = (): void => {
                logger.info(`[${isProduction ? 'PROD' : 'DEV'}] WebSocket connected successfully`);
                this.connectionState = ConnectionState.CONNECTED;
                this.reconnectAttempts = 0;
                this.lastPongTime = Date.now();
                this.startConnectionMonitor();

                // Notify connection status change
                if (this.connectionStatusHandler) {
                    this.connectionStatusHandler(true);
                }

                // Send initial protocol messages
                this.sendMessage({ type: 'requestInitialData' });
                this.sendMessage({ type: 'enableBinaryUpdates' });
            };

            this.ws.onerror = (event: Event): void => {
                const errorMsg = isProduction ? 
                    'WebSocket error in production (possible Cloudflare issue)' : 
                    'WebSocket error in development';
                logger.error(errorMsg, event);
                
                if (this.ws) {
                    logger.error('WebSocket readyState:', this.ws.readyState);
                    
                    if (isProduction) {
                        logger.error('Production debug info:', {
                            url: wsUrl,
                            protocol: window.location.protocol,
                            host: window.location.host,
                            hostname: window.location.hostname,
                            cloudflareRay: document.querySelector('meta[name="cf-ray"]')?.getAttribute('content')
                        });
                    }
                }
            };

            this.ws.onclose = (event: CloseEvent): void => {
                const envPrefix = isProduction ? '[PROD]' : '[DEV]';
                logger.warn(`${envPrefix} WebSocket closed with code ${event.code}: ${event.reason}`);
                
                if (isProduction && event.code === 1006) {
                    logger.error('Cloudflare connection dropped (code 1006). This might indicate a tunnel issue.');
                }
                
                // Notify connection status change
                if (this.connectionStatusHandler) {
                    this.connectionStatusHandler(false);
                }
                
                this.handleReconnect();
            };

            this.ws.onmessage = async (event: MessageEvent): Promise<void> => {
                try {
                    // Update last pong time for any successful message
                    this.lastPongTime = Date.now();

                    if (this.connectionState !== ConnectionState.CONNECTED || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                        logger.warn('WebSocket not connected, ignoring message');
                        return;
                    }

                    // Handle text messages (errors from server)
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
                        if (event.data.byteLength % NODE_POSITION_SIZE !== 0) {
                            logger.error('Invalid binary message length:', event.data.byteLength);
                            return;
                        }

                        const float32Array = new Float32Array(event.data);
                        const nodeCount = float32Array.length / FLOATS_PER_NODE;
                        const nodes: NodeData[] = [];

                        for (let i = 0; i < nodeCount; i++) {
                            const baseIndex = i * FLOATS_PER_NODE;
                            
                            // Validate float values
                            const values = float32Array.slice(baseIndex, baseIndex + FLOATS_PER_NODE);
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
                        } else {
                            logger.warn('No valid nodes in binary message');
                        }
                    }
                } catch (error) {
                    logger.error('Error processing WebSocket message:', error);
                    if (error instanceof Error) {
                        logger.error('Error details:', {
                            name: error.name,
                            message: error.message,
                            stack: error.stack
                        });
                    }
                }
            };
        } catch (error) {
            logger.error('Failed to setup WebSocket:', error);
            this.connectionState = ConnectionState.FAILED;
            
            if (this.connectionStatusHandler) {
                this.connectionStatusHandler(false);
            }
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = this.getReconnectDelay();
                
                logger.info(
                    `WebSocket setup failed, attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
                );
                
                this.connectionState = ConnectionState.RECONNECTING;
                
                this.reconnectTimeout = window.setTimeout(() => {
                    this.reconnectTimeout = null;
                    this.setupWebSocket();
                }, delay);
            }
        }
    }

    private handleReconnect(): void {
        const wasConnected = this.connectionState === ConnectionState.CONNECTED;
        this.connectionState = ConnectionState.DISCONNECTED;
        this.binaryMessageCallback = null;
        this.stopConnectionMonitor();
        
        if (this.reconnectTimeout !== null) {
            window.clearTimeout(this.reconnectTimeout);
        }
        
        if (this.reconnectAttempts < this.maxReconnectAttempts &&
            (wasConnected || this.reconnectAttempts === 0)) {
            
            this.reconnectAttempts++;
            const delay = this.getReconnectDelay();
            
            logger.info(
                `WebSocket connection closed, attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`
            );
            
            this.connectionState = ConnectionState.RECONNECTING;
            
            this.reconnectTimeout = window.setTimeout(() => {
                this.reconnectTimeout = null;
                this.setupWebSocket();
            }, delay);
        } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
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
            const float32Array = new Float32Array(validUpdates.length * FLOATS_PER_NODE);
            
            validUpdates.forEach((update, index) => {
                const baseIndex = index * FLOATS_PER_NODE;
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
        
        this.stopConnectionMonitor();
        
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
}
