import { createLogger } from '../core/logger';
import { buildWsUrl } from '../core/api';

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
        this.connect();
    }

    private connect(): void {
        try {
            const wsUrl = buildWsUrl();
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                logger.info('WebSocket connection established');
                this.reconnectAttempts = 0;
                this.startHeartbeat();
            };

            this.ws.onclose = () => {
                logger.info('WebSocket connection closed');
                this.handleDisconnect();
            };

            this.ws.onerror = (error) => {
                logger.error('WebSocket error:', error);
                this.handleDisconnect();
            };

            this.ws.onmessage = (event) => {
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
            };
        } catch (error) {
            logger.error('Error creating WebSocket connection:', error);
            this.handleDisconnect();
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
