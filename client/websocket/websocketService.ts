import { createLogger } from '../core/logger';
import { buildWsUrl, buildSettingsUrl } from '../core/api';
import { WebSocketSettings } from '../core/types';
import { snakeToCamelCase } from '../core/utils';

const logger = createLogger('WebSocketService');

type MessageHandler = (data: any) => void;

export class WebSocketService {
    private ws: WebSocket | null = null;
    private messageHandlers = new Map<string, MessageHandler>();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 3;
    private reconnectDelay = 5000; // 5 seconds
    private heartbeatInterval = 15000; // 15 seconds
    private heartbeatTimeout = 60000; // 60 seconds
    private lastHeartbeat: number | null = null;
    private heartbeatTimer: NodeJS.Timeout | null = null;
    private reconnectTimer: NodeJS.Timeout | null = null;

    constructor() {
        this.loadSettings().then(() => this.connect());
    }

    private async loadSettings(): Promise<void> {
        try {
            // Load each WebSocket setting individually
            const settings: Partial<WebSocketSettings> = {};
            const settingKeys = [
                'heartbeat-interval',
                'heartbeat-timeout',
                'max-reconnect-attempts',
                'reconnect-delay',
                'update-rate'
            ];

            await Promise.all(
                settingKeys.map(async (setting) => {
                    try {
                        const response = await fetch(buildSettingsUrl('websocket', setting));
                        if (!response.ok) {
                            throw new Error(`Failed to load setting ${setting}: ${response.statusText}`);
                        }
                        const data = await response.json();
                        const camelKey = snakeToCamelCase(setting.replace(/-/g, '_'));
                        (settings as any)[camelKey] = data.value;
                    } catch (error) {
                        logger.error(`Error loading WebSocket setting ${setting}:`, error);
                    }
                })
            );

            this.updateSettings(settings);
        } catch (error) {
            logger.error('Error loading WebSocket settings:', error);
        }
    }

    private updateSettings(settings: Partial<WebSocketSettings>): void {
        this.heartbeatInterval = settings.heartbeatInterval ?? 15000;
        this.heartbeatTimeout = settings.heartbeatTimeout ?? 60000;
        this.maxReconnectAttempts = settings.reconnectAttempts ?? 3;
        this.reconnectDelay = settings.reconnectDelay ?? 5000;
        
        logger.debug('Updated WebSocket settings:', {
            heartbeatInterval: this.heartbeatInterval,
            heartbeatTimeout: this.heartbeatTimeout,
            maxReconnectAttempts: this.maxReconnectAttempts,
            reconnectDelay: this.reconnectDelay
        });
    }

    private connect(): void {
        try {
            const wsUrl = buildWsUrl();
            this.ws = new WebSocket(wsUrl);
            
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
        this.reconnectAttempts = 0;
        this.startHeartbeat();
    }

    private onclose(): void {
        logger.info('WebSocket connection closed');
        this.handleDisconnect();
    }

    private onerror(error: Event): void {
        logger.error('WebSocket error:', error);
        this.handleDisconnect();
    }

    private onmessage(event: MessageEvent): void {
        try {
            const message = JSON.parse(event.data);
            if (message.type === 'heartbeat') {
                this.handleHeartbeat();
            } else {
                const handler = this.messageHandlers.get(message.type);
                if (handler) {
                    handler(message.data);
                }
            }
        } catch (error) {
            logger.error('Error handling WebSocket message:', error);
        }
    }

    private handleDisconnect(): void {
        this.stopHeartbeat();
        
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            logger.info(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
            }
            
            this.reconnectTimer = setTimeout(() => {
                this.connect();
            }, this.reconnectDelay);
        } else {
            logger.error('Max reconnection attempts reached');
        }
    }

    private startHeartbeat(): void {
        this.lastHeartbeat = Date.now();
        
        // Send heartbeat
        this.heartbeatTimer = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'heartbeat' }));
                
                // Check if we've missed too many heartbeats
                const timeSinceLastHeartbeat = Date.now() - (this.lastHeartbeat || 0);
                if (timeSinceLastHeartbeat > this.heartbeatTimeout) {
                    logger.warn('Heartbeat timeout, reconnecting...');
                    this.ws.close();
                }
            }
        }, this.heartbeatInterval);
    }

    private stopHeartbeat(): void {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    private handleHeartbeat(): void {
        this.lastHeartbeat = Date.now();
    }

    public onMessage(type: string, handler: MessageHandler): void {
        this.messageHandlers.set(type, handler);
    }

    public send(data: string): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(data);
        } else {
            logger.warn('Attempted to send message while WebSocket is not connected');
        }
    }

    public dispose(): void {
        this.stopHeartbeat();
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.messageHandlers.clear();
    }
}
