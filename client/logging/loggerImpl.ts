import { Logger } from '../core/types';

export class LoggerImpl implements Logger {
    debug(message: string, ...args: unknown[]): void {
        console.debug(message, ...args);
    }

    info(message: string, ...args: unknown[]): void {
        console.info(message, ...args);
    }

    warn(message: string, ...args: unknown[]): void {
        console.warn(message, ...args);
    }

    error(message: string, ...args: unknown[]): void {
        console.error(message, ...args);
    }

    log(message: string, ...args: unknown[]): void {
        console.log(message, ...args);
    }
}
