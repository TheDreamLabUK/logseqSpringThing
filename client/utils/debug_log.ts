// Debug logging levels
export enum LogLevel {
  ERROR = 'ERROR',
  WARN = 'WARN',
  DEBUG = 'DEBUG'
}

// Debug settings interface matching settings.toml
interface ClientDebugSettings {
  enabled: boolean;
  enable_websocket_debug: boolean;
  enable_data_debug: boolean;
  log_full_json: boolean;
}

/**
 * IMPORTANT: Debug Override Configuration
 * 
 * This override forces error-only logging regardless of settings.toml configuration.
 * This was added to prevent performance issues from excessive logging.
 * 
 * To re-enable full logging:
 * 1. Set ERROR_ONLY_OVERRIDE to false
 * 2. Ensure your settings.toml has appropriate debug settings
 * 3. Monitor performance impact of additional logging
 * 
 * Location: client/utils/debug_log.ts
 * Related: settings.toml client_debug section
 */
const ERROR_ONLY_OVERRIDE = true; // Set to false to restore normal debug levels

let debugSettings: ClientDebugSettings = {
  enabled: false,
  enable_websocket_debug: false,
  enable_data_debug: false,
  log_full_json: false
};

// Initialize debug settings
export function initDebugSettings(settings: Partial<ClientDebugSettings>) {
  if (ERROR_ONLY_OVERRIDE) {
    // When override is active, only allow error logging
    debugSettings = {
      enabled: false,
      enable_websocket_debug: false,
      enable_data_debug: false,
      log_full_json: false
    };
    console.info('[Debug] Error-only logging enforced by ERROR_ONLY_OVERRIDE');
  } else {
    debugSettings = { ...debugSettings, ...settings };
  }
}

// Format data for logging
function formatData(data: any): string {
  if (data instanceof ArrayBuffer) {
    return `Binary Data: ${data.byteLength} bytes`;
  }
  
  if (data instanceof Event) {
    return `Event: ${data.type}`;
  }
  
  if (data instanceof Error) {
    return `Error: ${data.message}`;
  }
  
  if (typeof data === 'object') {
    try {
      return JSON.stringify(data, null, 2);
    } catch {
      return String(data);
    }
  }
  
  return String(data);
}

// Base logging function
function log(level: LogLevel, context: string, message: string, data?: any) {
  // When override is active, only allow ERROR level
  if (ERROR_ONLY_OVERRIDE && level !== LogLevel.ERROR) {
    return;
  }

  const timestamp = new Date().toISOString();
  const prefix = `[${level} ${context} ${timestamp}]`;
  
  switch (level) {
    case LogLevel.ERROR:
      console.error(`${prefix} ${message}`, data ? '\n' + formatData(data) : '');
      break;
    case LogLevel.WARN:
      if (debugSettings.enabled) {
        console.warn(`${prefix} ${message}`, data ? '\n' + formatData(data) : '');
      }
      break;
    case LogLevel.DEBUG:
      if (debugSettings.enabled || 
          (context === 'WS' && debugSettings.enable_websocket_debug) ||
          (context === 'DATA' && debugSettings.enable_data_debug)) {
        console.debug(`${prefix} ${message}`, data ? '\n' + formatData(data) : '');
      }
      break;
  }
}

// Exported logging functions
export const logError = (message: string, data?: any) => log(LogLevel.ERROR, 'APP', message, data);
export const logWarn = (message: string, data?: any) => log(LogLevel.WARN, 'APP', message, data);
export const logDebug = (message: string, data?: any) => log(LogLevel.DEBUG, 'APP', message, data);

// Context-specific logging
export const logWebsocket = (message: string, data?: any) => log(LogLevel.DEBUG, 'WS', message, data);
export const logData = (message: string, data?: any) => log(LogLevel.DEBUG, 'DATA', message, data);

// JSON specific logging
export function logJson(data: any) {
  // Skip JSON logging when override is active
  if (ERROR_ONLY_OVERRIDE) return;

  if (debugSettings.log_full_json) {
    try {
      const formatted = JSON.stringify(data, null, 2);
      log(LogLevel.DEBUG, 'JSON', 'Full JSON data:', formatted);
    } catch (error) {
      log(LogLevel.ERROR, 'JSON', 'Failed to stringify JSON', error);
    }
  }
}

// Get current debug settings
export function getDebugSettings(): Readonly<ClientDebugSettings> {
  return { ...debugSettings };
}

// Reset debug settings to default
export function resetDebugSettings() {
  if (ERROR_ONLY_OVERRIDE) {
    // When override is active, maintain error-only configuration
    debugSettings = {
      enabled: false,
      enable_websocket_debug: false,
      enable_data_debug: false,
      log_full_json: false
    };
  } else {
    debugSettings = {
      enabled: false,
      enable_websocket_debug: false,
      enable_data_debug: false,
      log_full_json: false
    };
  }
}
