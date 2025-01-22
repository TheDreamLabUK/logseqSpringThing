import { createLogger } from '../utils/logger';
import { EventEmitter } from '../utils/eventEmitter';

interface WebSocketEvents {
    connected: boolean;
    message: any;
    error: Event;
}

export class WebSocketService extends EventEmitter<WebSocketEvents> {
    private static instance: WebSocketService;
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private readonly maxReconnectAttempts = 5;
    private readonly logger = createLogger('WebSocketService');

    private constructor() {
        super();
    }

    public static getInstance(): WebSocketService {
        if (!WebSocketService.instance) {
            WebSocketService.instance = new WebSocketService();
        }
        return WebSocketService.instance;
    }

    public connect(): void {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
            this.logger.warn('WebSocket already connected or connecting');
            return;
        }

        const url = this.getWebSocketUrl();
        this.ws = new WebSocket(url);
        this.setupEventHandlers();
    }

    private getWebSocketUrl(): string {
        // Always use secure WebSocket when going through Cloudflare
        const protocol = 'wss:';
        const host = window.location.host;
        return `${protocol}//${host}/wss`;
    }

    private setupEventHandlers(): void {
        if (!this.ws) return;

        this.ws.onopen = () => {
            this.logger.info('WebSocket connected');
            this.reconnectAttempts = 0;
            this.emit('connected', true);
        };

        this.ws.onclose = () => {
            this.logger.warn('WebSocket closed');
            this.emit('connected', false);
            this.reconnect();
        };

        this.ws.onerror = (event: Event) => {
            this.handleError(event);
        };

        this.ws.onmessage = (event: MessageEvent) => {
            this.emit('message', event.data);
        };
    }

    private handleError(event: Event): void {
        this.logger.error('WebSocket error:', event);
        this.emit('error', event);
        if (this.ws?.readyState === WebSocket.CLOSED) {
            this.reconnect();
        }
    }

    private reconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            this.logger.info(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connect(), delay);
        } else {
            this.logger.error('Max reconnection attempts reached');
        }
    }

    public disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

export const webSocketService = WebSocketService.getInstance(); 