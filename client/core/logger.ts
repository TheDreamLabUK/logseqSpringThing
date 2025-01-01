import { Logger } from './types';

interface LoggerOptions {
    namespace?: string;
    level?: 'debug' | 'info' | 'warn' | 'error';
}

export class LoggerImpl implements Logger {
    private namespace: string;
    private level: string;
    private static enabled = true;

    constructor(options: LoggerOptions | string = {}) {
        if (typeof options === 'string') {
            this.namespace = options;
            this.level = 'info';
        } else {
            this.namespace = options.namespace || 'default';
            this.level = options.level || 'info';
        }
    }

    private formatMessage(level: string, message: string): string {
        return `[${this.namespace}] [${level}] ${message}`;
    }

    debug(message: string, ...args: unknown[]): void {
        if (!LoggerImpl.enabled) return;
        if (this.level === 'debug') {
            console.debug(this.formatMessage('DEBUG', message), ...args);
        }
    }

    info(message: string, ...args: unknown[]): void {
        if (!LoggerImpl.enabled) return;
        if (this.level === 'debug' || this.level === 'info') {
            console.info(this.formatMessage('INFO', message), ...args);
        }
    }

    warn(message: string, ...args: unknown[]): void {
        if (!LoggerImpl.enabled) return;
        if (this.level !== 'error') {
            console.warn(this.formatMessage('WARN', message), ...args);
        }
    }

    error(message: string, ...args: unknown[]): void {
        if (!LoggerImpl.enabled) return;
        console.error(this.formatMessage('ERROR', message), ...args);
    }

    log(message: string, ...args: unknown[]): void {
        if (!LoggerImpl.enabled) return;
        console.log(this.formatMessage('LOG', message), ...args);
    }

    static setEnabled(enabled: boolean): void {
        LoggerImpl.enabled = enabled;
    }

    static isEnabled(): boolean {
        return LoggerImpl.enabled;
    }
}

export function createLogger(options?: LoggerOptions | string): Logger {
    return new LoggerImpl(options);
}

// Global logging controls
export function setLoggingEnabled(enabled: boolean): void {
    LoggerImpl.setEnabled(enabled);
}

export function isLoggingEnabled(): boolean {
    return LoggerImpl.isEnabled();
}

// Create core logger instance
export const logger = createLogger({ namespace: 'core' });
