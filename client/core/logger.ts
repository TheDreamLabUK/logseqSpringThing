let debugEnabled = false;
let logFullJson = false;

export interface Logger {
    debug: (...args: any[]) => void;
    log: (...args: any[]) => void;
    info: (...args: any[]) => void;
    warn: (...args: any[]) => void;
    error: (...args: any[]) => void;
}

export function setDebugEnabled(enabled: boolean, fullJson: boolean = false): void {
    debugEnabled = enabled;
    logFullJson = fullJson;
}

export function createLogger(context: string): Logger {
    const prefix = `[${context}]`;
    
    const formatArgs = (args: any[]): any[] => {
        if (logFullJson) {
            return args.map(arg => 
                typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg
            );
        }
        return args;
    };
    
    return {
        debug: (...args: any[]): void => {
            if (debugEnabled) {
                console.debug(prefix, ...formatArgs(args));
            }
        },
        log: (...args: any[]): void => {
            console.log(prefix, ...formatArgs(args));
        },
        info: (...args: any[]): void => {
            console.info(prefix, ...formatArgs(args));
        },
        warn: (...args: any[]): void => {
            console.warn(prefix, ...formatArgs(args));
        },
        error: (...args: any[]): void => {
            console.error(prefix, ...formatArgs(args));
        }
    };
}

// Create core logger instance
export const logger = createLogger('core');
