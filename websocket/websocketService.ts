import { Logger } from '../utils/logger';
import { ConnectionState } from './connectionState';

const log = new Logger('WebSocketService');

interface WebSocketSettings {
    url: string;
    reconnectDelay: number;
    maxReconnectAttempts: number;
}

export class WebSocketService {
    private static instance: WebSocketService | null = null;
    private ws: WebSocket | null = null;
    private settings: WebSocketSettings;
    
    private buildWsUrl(): string {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }

    public async initialize(): Promise<void> {
        await this.loadWebSocketSettings();
        
        const wsUrl = this.buildWsUrl();
        log.debug('Initializing WebSocket connection to:', wsUrl);
        
        try {
            this.ws = new WebSocket(wsUrl);
            this.setupWebSocketHandlers();
        } catch (error) {
            log.error('Failed to initialize WebSocket:', error);
            throw error;
        }
    }

    private setupWebSocketHandlers(): void {
        if (!this.ws) return;

        this.ws.binaryType = 'arraybuffer';
        
        this.ws.onopen = () => {
            this.connectionState = ConnectionState.Connected;
            this.connectionStatusHandler?.(ConnectionState.Connected);
            this.reconnectAttempts = 0;
            log.info('WebSocket connected');
        };

        this.ws.onclose = () => {
            this.handleClose();
        };

        this.ws.onerror = (error) => {
            this.handleError(error);
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event);
        };
    }

    private async loadWebSocketSettings(): Promise<void> {
        try {
            const response = await fetch('/api/settings/websocket');
            if (!response.ok) {
                throw new Error(`Failed to load WebSocket settings: ${response.statusText}`);
            }
            const settings = await response.json();
            this.settings = {
                ...this.defaultSettings,
                ...settings
            };
        } catch (error) {
            log.error('Failed to load WebSocket settings:', error);
            this.settings = this.defaultSettings;
        }
    }

    private handleError(error: Error): void {
        log.error('WebSocket error:', error);
        this.connectionState = ConnectionState.Error;
        this.connectionStatusHandler?.(ConnectionState.Error);
        
        // Attempt reconnection with exponential backoff
        const backoff = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        this.reconnectAttempts++;
        
        setTimeout(() => {
            this.connect();
        }, backoff + Math.random() * 1000); // Add jitter
    }

    public close(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connectionState = ConnectionState.Closed;
        this.connectionStatusHandler?.(ConnectionState.Closed);
    }

    public dispose(): void {
        this.close();
        this.connectionStatusHandler = null;
        this.binaryUpdateHandler = null;
    }
} 