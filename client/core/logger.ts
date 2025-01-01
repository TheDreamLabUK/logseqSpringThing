import { Logger, LogLevel, LoggerOptions } from './types';
import { Settings } from '../types/settings';

export class LoggerImpl implements Logger {
    private static instance: LoggerImpl;
    private static settings: Settings;
    private namespace: string;
    private level: LogLevel;
    private jsonFormatting: boolean;

    constructor(options: LoggerOptions = {}) {
        this.namespace = options.namespace || 'default';
        this.level = options.level || 'info';
        this.jsonFormatting = options.enableJsonFormatting || false;
    }

    private static getInstance(): LoggerImpl {
        if (!LoggerImpl.instance) {
            LoggerImpl.instance = new LoggerImpl();
        }
        return LoggerImpl.instance;
    }

    public static setSettings(settings: Settings): void {
        LoggerImpl.settings = settings;
        const instance = LoggerImpl.getInstance();
        if (settings?.system?.debug) {
            instance.setLevel(settings.system.debug.logLevel as LogLevel);
            instance.setJsonFormatting(settings.system.debug.logFullJson);
        }
    }

    private shouldLog(level: LogLevel): boolean {
        const levels: LogLevel[] = ['error', 'warn', 'info', 'debug', 'trace'];
        const currentLevelIndex = levels.indexOf(this.level);
        const targetLevelIndex = levels.indexOf(level);
        return targetLevelIndex <= currentLevelIndex;
    }

    private formatMessage(level: LogLevel, message: string): string {
        return `[${this.namespace}] [${level.toUpperCase()}] ${message}`;
    }

    private formatArgs(args: unknown[]): unknown[] {
        if (!this.jsonFormatting) return args;
        return args.map(arg => 
            typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg
        );
    }

    debug(message: string, ...args: unknown[]): void {
        if (this.shouldLog('debug')) {
            console.debug(this.formatMessage('debug', message), ...this.formatArgs(args));
        }
    }

    info(message: string, ...args: unknown[]): void {
        if (this.shouldLog('info')) {
            console.info(this.formatMessage('info', message), ...this.formatArgs(args));
        }
    }

    warn(message: string, ...args: unknown[]): void {
        if (this.shouldLog('warn')) {
            console.warn(this.formatMessage('warn', message), ...this.formatArgs(args));
        }
    }

    error(message: string, ...args: unknown[]): void {
        if (this.shouldLog('error')) {
            console.error(this.formatMessage('error', message), ...this.formatArgs(args));
        }
    }

    trace(message: string, ...args: unknown[]): void {
        if (this.shouldLog('trace')) {
            console.debug(this.formatMessage('trace', message), ...this.formatArgs(args));
        }
    }

    log(level: LogLevel, message: string, ...args: unknown[]): void {
        switch (level) {
            case 'error': this.error(message, ...args); break;
            case 'warn': this.warn(message, ...args); break;
            case 'info': this.info(message, ...args); break;
            case 'debug': this.debug(message, ...args); break;
            case 'trace': this.trace(message, ...args); break;
        }
    }

    setLevel(level: LogLevel): void {
        this.level = level;
    }

    getLevel(): LogLevel {
        return this.level;
    }

    setJsonFormatting(enabled: boolean): void {
        this.jsonFormatting = enabled;
    }
}

export function createLogger(options?: LoggerOptions | string): Logger {
    if (typeof options === 'string') {
        options = { namespace: options };
    }
    return new LoggerImpl(options);
}

// Global logger instance
export const logger = createLogger({ namespace: 'core' });
