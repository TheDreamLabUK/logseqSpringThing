import { SettingsStore } from '../state/SettingsStore';

export type LogLevel = 'error' | 'warn' | 'info' | 'debug' | 'trace';

export interface LoggerConfig {
    level: LogLevel;
    enabled: boolean;
}

let globalConfig: LoggerConfig = {
    level: 'debug',
    enabled: true
};

export function configureLogger(config: Partial<LoggerConfig>) {
    globalConfig = { ...globalConfig, ...config };
    console.log('Logger configured with:', globalConfig);  // This will always show
}

export function createLogger(name: string) {
    const prefix = `[${name}]`;
    
    return {
        error: (...args: any[]) => {
            if (!globalConfig.enabled) return;
            console.error(prefix, ...args);
        },
        warn: (...args: any[]) => {
            if (!globalConfig.enabled || !isLevelEnabled('warn')) return;
            console.warn(prefix, ...args);
        },
        info: (...args: any[]) => {
            if (!globalConfig.enabled || !isLevelEnabled('info')) return;
            console.info(prefix, ...args);
        },
        debug: (...args: any[]) => {
            if (!globalConfig.enabled || !isLevelEnabled('debug')) return;
            console.debug(prefix, ...args);
        },
        trace: (...args: any[]) => {
            if (!globalConfig.enabled || !isLevelEnabled('trace')) return;
            console.trace(prefix, ...args);
        }
    };
}

function isLevelEnabled(level: LogLevel): boolean {
    const levels: LogLevel[] = ['error', 'warn', 'info', 'debug', 'trace'];
    const configIndex = levels.indexOf(globalConfig.level);
    const levelIndex = levels.indexOf(level);
    return levelIndex <= configIndex;
}

// Initialize logger with settings
export function initializeLogger() {
    const settings = SettingsStore.getInstance().get('system.debug');
    configureLogger({
        level: settings?.logLevel || 'debug',
        enabled: settings?.enabled ?? true
    });
}

export function checkLoggerConfig() {
    console.log('Current logger configuration:', {
        level: globalConfig.level,
        enabled: globalConfig.enabled
    });
    
    const testLogger = createLogger('TEST');
    testLogger.error('Test error message');
    testLogger.warn('Test warning message');
    testLogger.info('Test info message');
    testLogger.debug('Test debug message');
    testLogger.trace('Test trace message');
} 