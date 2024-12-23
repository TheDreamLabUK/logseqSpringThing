import { createLogger } from '../core/logger';
import { buildWsUrl, buildSettingsUrl } from '../core/api';
import { WebSocketSettings } from '../core/types';
import { ConnectionState, MessageType, WebSocketMessage } from '../types/websocket';

const logger = createLogger('WebSocketService');
const VALUES_PER_NODE = 6; // 3 for position, 3 for velocity

type BinaryMessageHandler = (positions: Float32Array, velocities: Float32Array) => void;

export class WebSocketService {
    private ws: WebSocket | null = null;
    private binaryHandler: BinaryMessageHandler | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 3;
    private reconnectDelay = 5000;
    private heartbeatInterval = 30000; // 30s to match server
    private heartbeatTimeout = 3600000; // 1h to match server
    private lastPingTime: number | null = null;
    private pingTimer: NodeJS.Timeout | null = null;
    private reconnectTimer: NodeJS.Timeout | null = null;
    private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
    private settingsLoaded = false;

    constructor() {
        this.initialize();
    }

    private async initialize(): Promise<void> {
        try {
            await this.loadSettings();
            this.settingsLoaded = true;
            await this.connect();
        } catch (error) {
            logger.error('Failed to initialize WebSocket service:', error);
        }
    }

    private async loadSettings(): Promise<void> {
        try {
            const response = await fetch(buildSettingsUrl('websocket'));
            if (!response.ok) {
                throw new Error(`Failed to load WebSocket settings: ${response.statusText}`);
            }
            const data = await response.json();
            if (data.success && data.settings) {
                this.updateSettings(data.settings);
            } else {
                throw new Error('Invalid settings response format');
            }
        } catch (error) {
            logger.error('Error loading WebSocket settings:', error);
            throw error;
        }
    }

    private updateSettings(settings: Partial<WebSocketSettings>): void {
        if (settings.heartbeatInterval) this.heartbeatInterval = settings.heartbeatInterval * 1000;
        if (settings.heartbeatTimeout) this.heartbeatTimeout = settings.heartbeatTimeout * 1000;
        if (settings.reconnectAttempts) this.maxReconnectAttempts = settings.reconnectAttempts;
        if (settings.reconnectDelay) this.reconnectDelay = settings.reconnectDelay;
        
        logger.debug('Updated WebSocket settings:', settings);
    }

    private async connect(): Promise<void> {
        if (this.connectionState === ConnectionState.CONNECTING || 
            this.connectionState === ConnectionState.CONNECTED) {
            return;
        }

        try {
            this.connectionState = ConnectionState.CONNECTING;
            const wsUrl = buildWsUrl();
            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';
            
            this.ws.onopen = this.onopen.bind(this);
            this.ws.onclose = this.onclose.bind(this);
            this.ws.onerror = this.onerror.bind(this);
            this.ws.onmessage = this.onmessage.bind(this);
            
            logger.debug('Attempting WebSocket connection to:', wsUrl);
        } catch (error) {
            logger.error('Error creating WebSocket connection:', error);
            this.handleDisconnect();
        }
    }

    private onopen(): void {
        logger.info('WebSocket connection established');
        this.connectionState = ConnectionState.CONNECTED;
        this.reconnectAttempts = 0;
        this.startPing();
    }

    private onclose(): void {
        logger.info('WebSocket connection closed');
        this.connectionState = ConnectionState.DISCONNECTED;
        this.handleDisconnect();
    }

    private onerror(error: Event): void {
        logger.error('WebSocket error:', error);
        this.handleDisconnect();
    }

    private onmessage(event: MessageEvent): void {
        try {
            // Handle binary messages (position/velocity updates)
            if (event.data instanceof ArrayBuffer) {
                if (this.binaryHandler) {
                    const float32Array = new Float32Array(event.data);
                    const nodeCount = float32Array.length / VALUES_PER_NODE;
                    
                    // Split the array into positions and velocities
                    const positions = new Float32Array(nodeCount * 3);
                    const velocities = new Float32Array(nodeCount * 3);
                    
                    for (let i = 0; i < nodeCount; i++) {
                        // Copy position (x, y, z)
                        positions[i * 3] = float32Array[i * VALUES_PER_NODE];
                        positions[i * 3 + 1] = float32Array[i * VALUES_PER_NODE + 1];
                        positions[i * 3 + 2] = float32Array[i * VALUES_PER_NODE + 2];
                        
                        // Copy velocity (vx, vy, vz)
                        velocities[i * 3] = float32Array[i * VALUES_PER_NODE + 3];
                        velocities[i * 3 + 1] = float32Array[i * VALUES_PER_NODE + 4];
                        velocities[i * 3 + 2] = float32Array[i * VALUES_PER_NODE + 5];
                    }
                    
                    this.binaryHandler(positions, velocities);
                }
                return;
            }

            // Handle ping/pong messages
            const message = JSON.parse(event.data) as WebSocketMessage;
            if (message.type === MessageType.PONG) {
                this.lastPingTime = Date.now();
            }
        } catch (error) {
            logger.error('Error handling WebSocket message:', error);
        }
    }

    private handleDisconnect(): void {
        this.stopPing();
        
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.connectionState = ConnectionState.RECONNECTING;
            logger.info(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
            }
            
            this.reconnectTimer = setTimeout(() => {
                this.connect();
            }, this.reconnectDelay);
        } else {
            logger.error('Max reconnection attempts reached');
            this.connectionState = ConnectionState.DISCONNECTED;
        }
    }

    private startPing(): void {
        this.lastPingTime = Date.now();
        
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
        }
        
        this.pingTimer = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                const pingMessage = {
                    type: MessageType.PING,
                    timestamp: Date.now()
                };
                this.ws.send(JSON.stringify(pingMessage));
                
                const timeSinceLastPing = Date.now() - (this.lastPingTime || 0);
                if (timeSinceLastPing > this.heartbeatTimeout) {
                    logger.warn('Ping timeout, reconnecting...');
                    this.ws.close();
                }
            }
        }, this.heartbeatInterval);
    }

    private stopPing(): void {
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
        this.lastPingTime = null;
    }

    /**
     * Register handler for binary position/velocity updates
     */
    public onBinaryMessage(handler: BinaryMessageHandler): void {
        this.binaryHandler = handler;
    }

    /**
     * Send binary position update
     */
    public sendBinaryUpdate(position: [number, number, number], velocity: [number, number, number]): void {
        if (!this.settingsLoaded || this.ws?.readyState !== WebSocket.OPEN) {
            return;
        }

        const buffer = new ArrayBuffer(VALUES_PER_NODE * 4); // 6 floats * 4 bytes
        const view = new Float32Array(buffer);
        
        // Position [f32; 3]
        view[0] = position[0];
        view[1] = position[1];
        view[2] = position[2];
        
        // Velocity [f32; 3]
        view[3] = velocity[0];
        view[4] = velocity[1];
        view[5] = velocity[2];
        
        this.ws.send(buffer);
    }

    public getConnectionState(): ConnectionState {
        return this.connectionState;
    }

    public dispose(): void {
        this.stopPing();
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.binaryHandler = null;
        this.connectionState = ConnectionState.DISCONNECTED;
        this.settingsLoaded = false;
    }
}
