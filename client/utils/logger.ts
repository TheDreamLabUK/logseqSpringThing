export interface Logger {
    log: (message: string, ...args: any[]) => void;
    error: (message: string, ...args: any[]) => void;
    warn: (message: string, ...args: any[]) => void;
    info: (message: string, ...args: any[]) => void;
    debug: (message: string, ...args: any[]) => void;
}

// Track debug state
let debugEnabled = true; // Start with debug enabled

export function setDebugEnabled(enabled: boolean): void {
    debugEnabled = enabled;
}

export function createLogger(context: string): Logger {
    const prefix = `[${context}]`;

    return {
        // Always log errors and warnings
        log: (message: string, ...args: any[]) => console.log(`${prefix} ${message}`, ...args),
        error: (message: string, ...args: any[]) => console.error(`${prefix} ${message}`, ...args),
        warn: (message: string, ...args: any[]) => console.warn(`${prefix} ${message}`, ...args),
        
        // Debug and info respect debug state
        info: (message: string, ...args: any[]) => {
            if (debugEnabled) {
                console.info(`${prefix} ${message}`, ...args);
            }
        },
        debug: (message: string, ...args: any[]) => {
            if (debugEnabled) {
                console.debug(`${prefix} ${message}`, ...args);
            }
        }
    };
}
