export class Logger {
    constructor(private context: string) {}

    debug(message: string, ...args: any[]): void {
        console.debug(`[${this.context}] ${message}`, ...args);
    }

    info(message: string, ...args: any[]): void {
        console.info(`[${this.context}] ${message}`, ...args);
    }

    warn(message: string, ...args: any[]): void {
        console.warn(`[${this.context}] ${message}`, ...args);
    }

    error(message: string, ...args: any[]): void {
        console.error(`[${this.context}] ${message}`, ...args);
    }
} 