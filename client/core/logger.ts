import { Settings } from '../types/settings';

let loggingEnabled = true;

export enum LogLevel {
    ERROR = 0,
    WARN = 1,
    INFO = 2,
    DEBUG = 3
}

interface LoggerConfig {
    level: LogLevel;
    namespace: string;
    timestamp: boolean;
    enabled: boolean;
}

const defaultConfig: LoggerConfig = {
    level: LogLevel.INFO,
    namespace: 'app',
    timestamp: true,
    enabled: true
};

export class LoggerImpl {
    private config: LoggerConfig;
    private logBuffer: string[] = [];
    private readonly MAX_BUFFER_SIZE = 1000;
    private static settings: Settings;

    constructor(config: Partial<LoggerConfig> = {}) {
        this.config = { ...defaultConfig, ...config };
    }

    static setSettings(settings: Settings): void {
        LoggerImpl.settings = settings;
    }

    private get isDebugEnabled(): boolean {
        return LoggerImpl.settings?.system?.debug?.enabled ?? false;
    }

    private get shouldLogFullJson(): boolean {
        return LoggerImpl.settings?.system?.debug?.logFullJson ?? false;
    }

    private formatMessage(level: string, message: string, args: unknown[]): string {
        const timestamp = this.config.timestamp ? `[${new Date().toISOString()}] ` : '';
        const namespace = `[${this.config.namespace}] `;
        const formattedArgs = args.map(arg => 
            arg instanceof Error ? arg.stack || arg.message : 
            typeof arg === 'object' ? JSON.stringify(arg, null, this.shouldLogFullJson ? 2 : null) : String(arg)
        ).join(' ');

        return `${timestamp}${level} ${namespace}${message} ${formattedArgs}`.trim();
    }

    private addToBuffer(message: string): void {
        this.logBuffer.push(message);
        if (this.logBuffer.length > this.MAX_BUFFER_SIZE) {
            this.logBuffer.shift();
        }
    }

    private writeToOutput(message: string, level: 'log' | 'error' | 'warn' | 'info' = 'log'): void {
        if (!this.config.enabled || !loggingEnabled) {
            return;
        }

        if (!this.isDebugEnabled && level === 'log') {
            return;
        }

        switch (level) {
            case 'error':
                console.error(message);
                break;
            case 'warn':
                console.warn(message);
                break;
            case 'info':
                console.info(message);
                break;
            default:
                if (this.isDebugEnabled) {
                    console.log(message);
                }
        }
    }

    debug(message: string, ...args: unknown[]): void {
        if (this.isDebugEnabled && this.config.level >= LogLevel.DEBUG) {
            const formattedMessage = this.formatMessage('DEBUG', message, args);
            this.addToBuffer(formattedMessage);
            this.writeToOutput(formattedMessage);
        }
    }

    log(message: string, ...args: unknown[]): void {
        if (this.config.level >= LogLevel.INFO) {
            const formattedMessage = this.formatMessage('INFO', message, args);
            this.addToBuffer(formattedMessage);
            this.writeToOutput(formattedMessage, 'info');
        }
    }

    info(message: string, ...args: unknown[]): void {
        if (this.config.level >= LogLevel.INFO) {
            const formattedMessage = this.formatMessage('INFO', message, args);
            this.addToBuffer(formattedMessage);
            this.writeToOutput(formattedMessage, 'info');
        }
    }

    warn(message: string, ...args: unknown[]): void {
        if (this.config.level >= LogLevel.WARN) {
            const formattedMessage = this.formatMessage('WARN', message, args);
            this.addToBuffer(formattedMessage);
            this.writeToOutput(formattedMessage, 'warn');
        }
    }

    error(message: string, ...args: unknown[]): void {
        if (this.config.level >= LogLevel.ERROR) {
            const formattedMessage = this.formatMessage('ERROR', message, args);
            this.addToBuffer(formattedMessage);
            this.writeToOutput(formattedMessage, 'error');
        }
    }

    getLogHistory(): string[] {
        return [...this.logBuffer];
    }

    clearLogHistory(): void {
        this.logBuffer = [];
    }

    setEnabled(enabled: boolean): void {
        this.config.enabled = enabled;
    }

    isEnabled(): boolean {
        return this.config.enabled && loggingEnabled;
    }

    setLevel(level: LogLevel): void {
        this.config.level = level;
    }

    getLevel(): LogLevel {
        return this.config.level;
    }
}

// Global logging controls
export function setLoggingEnabled(enabled: boolean): void {
    loggingEnabled = enabled;
}

export function isLoggingEnabled(): boolean {
    return loggingEnabled;
}

export function createLogger(context: string): LoggerImpl {
    return new LoggerImpl({ namespace: context });
}

// Create core logger instance
export const logger = createLogger('core');
