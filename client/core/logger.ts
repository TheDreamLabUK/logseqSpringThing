let debugEnabled = false;

export interface Logger {
    debug: (...args: any[]) => void;
    log: (...args: any[]) => void;
    info: (...args: any[]) => void;
    warn: (...args: any[]) => void;
    error: (...args: any[]) => void;
}

export function setDebugEnabled(enabled: boolean): void {
    debugEnabled = enabled;
}

export function createLogger(context: string): Logger {
    const prefix = `[${context}]`;
    
    return {
        debug: (...args: any[]): void => {
            if (debugEnabled) {
                console.debug(prefix, ...args);
            }
        },
        log: (...args: any[]): void => {
            console.log(prefix, ...args);
        },
        info: (...args: any[]): void => {
            console.info(prefix, ...args);
        },
        warn: (...args: any[]): void => {
            console.warn(prefix, ...args);
        },
        error: (...args: any[]): void => {
            console.error(prefix, ...args);
        }
    };
}

// Create core logger instance
export const logger = createLogger('core');
