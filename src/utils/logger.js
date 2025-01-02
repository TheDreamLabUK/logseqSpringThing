// Debug levels
export const LogLevel = {
    DEBUG: 'debug',
    INFO: 'info',
    WARN: 'warn',
    ERROR: 'error'
};

class Logger {
    constructor() {
        this.isDebugEnabled = process.env.NODE_ENV !== 'production';
        this.debugNamespaces = new Set();
    }

    // Enable debug for specific namespaces
    enableDebug(namespace) {
        this.debugNamespaces.add(namespace);
    }

    // Disable debug for specific namespaces
    disableDebug(namespace) {
        this.debugNamespaces.delete(namespace);
    }

    // Format message with timestamp and namespace
    formatMessage(namespace, level, message, ...args) {
        const timestamp = new Date().toISOString();
        const formattedMessage = typeof message === 'string' ? message : JSON.stringify(message);
        return `[${timestamp}] [${namespace}] [${level.toUpperCase()}] ${formattedMessage}`;
    }

    // Core logging function
    log(namespace, level, message, ...args) {
        if (!this.isDebugEnabled && level === LogLevel.DEBUG) return;
        if (!this.debugNamespaces.has(namespace) && level === LogLevel.DEBUG) return;

        const formattedMessage = this.formatMessage(namespace, level, message);
        
        switch (level) {
            case LogLevel.DEBUG:
                console.debug(formattedMessage, ...args);
                break;
            case LogLevel.INFO:
                console.info(formattedMessage, ...args);
                break;
            case LogLevel.WARN:
                console.warn(formattedMessage, ...args);
                break;
            case LogLevel.ERROR:
                console.error(formattedMessage, ...args);
                break;
        }
    }

    // Convenience methods
    debug(namespace, message, ...args) {
        this.log(namespace, LogLevel.DEBUG, message, ...args);
    }

    info(namespace, message, ...args) {
        this.log(namespace, LogLevel.INFO, message, ...args);
    }

    warn(namespace, message, ...args) {
        this.log(namespace, LogLevel.WARN, message, ...args);
    }

    error(namespace, message, ...args) {
        this.log(namespace, LogLevel.ERROR, message, ...args);
    }

    // Performance logging
    time(namespace, label) {
        if (this.isDebugEnabled) {
            console.time(`[${namespace}] ${label}`);
        }
    }

    timeEnd(namespace, label) {
        if (this.isDebugEnabled) {
            console.timeEnd(`[${namespace}] ${label}`);
        }
    }

    // Group logging for better visualization
    group(namespace, label) {
        if (this.isDebugEnabled) {
            console.group(`[${namespace}] ${label}`);
        }
    }

    groupEnd() {
        if (this.isDebugEnabled) {
            console.groupEnd();
        }
    }
}

// Create singleton instance
const logger = new Logger();

// Enable debug for core components by default
logger.enableDebug('webxr');
logger.enableDebug('graph');
logger.enableDebug('websocket');
logger.enableDebug('ragflow');
logger.enableDebug('perplexity');
logger.enableDebug('audio');

export default logger;
