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
  log_binary_headers: boolean;
  log_full_json: boolean;
}

let debugSettings: ClientDebugSettings = {
  enabled: false,
  enable_websocket_debug: false,
  enable_data_debug: false,
  log_binary_headers: false,
  log_full_json: false
};

// Initialize debug settings
export function initDebugSettings(settings: Partial<ClientDebugSettings>) {
  debugSettings = { ...debugSettings, ...settings };
  console.info('Client debug settings initialized:', debugSettings);
}

// Format data for logging
function formatData(data: any): string {
  if (data instanceof ArrayBuffer) {
    const nodeCount = data.byteLength / 24;
    return `Binary Data: ${nodeCount} nodes, ${data.byteLength} bytes`;
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

// Binary data specific logging
export function logBinaryHeader(data: ArrayBuffer) {
  if (debugSettings.log_binary_headers) {
    const header = new Uint8Array(data.slice(0, 16));
    const hexHeader = Array.from(header)
      .map(b => b.toString(16).padStart(2, '0'))
      .join(' ');
    log(LogLevel.DEBUG, 'BINARY', `Header: ${hexHeader}`);
  }
}

// JSON specific logging
export function logJson(data: any) {
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

// Toggle specific debug features
export function toggleDebugFeature(feature: keyof ClientDebugSettings): boolean {
  if (feature in debugSettings) {
    debugSettings[feature] = !debugSettings[feature];
    console.info(`Debug feature '${feature}' ${debugSettings[feature] ? 'enabled' : 'disabled'}`);
    return debugSettings[feature];
  }
  return false;
}

// Reset debug settings to default
export function resetDebugSettings() {
  debugSettings = {
    enabled: false,
    enable_websocket_debug: false,
    enable_data_debug: false,
    log_binary_headers: false,
    log_full_json: false
  };
  console.info('Debug settings reset to defaults');
}
