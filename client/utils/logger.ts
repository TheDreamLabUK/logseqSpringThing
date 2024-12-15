export interface Logger {
    log: (message: string, ...args: any[]) => void;
    error: (message: string, ...args: any[]) => void;
    warn: (message: string, ...args: any[]) => void;
    info: (message: string, ...args: any[]) => void;
    debug: (message: string, ...args: any[]) => void;
}

export function createLogger(context: string): Logger {
    const prefix = `[${context}]`;
    return {
        log: (message: string, ...args: any[]) => console.log(`${prefix} ${message}`, ...args),
        error: (message: string, ...args: any[]) => console.error(`${prefix} ${message}`, ...args),
        warn: (message: string, ...args: any[]) => console.warn(`${prefix} ${message}`, ...args),
        info: (message: string, ...args: any[]) => console.info(`${prefix} ${message}`, ...args),
        debug: (message: string, ...args: any[]) => console.debug(`${prefix} ${message}`, ...args)
    };
}
