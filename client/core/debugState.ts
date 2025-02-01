import { SettingsStore } from '../state/SettingsStore';

export interface DebugState {
    enabled: boolean;
    logFullJson: boolean;
    enableDataDebug: boolean;
    enableWebsocketDebug: boolean;
    logBinaryHeaders: boolean;
}

class DebugStateManager {
    private static instance: DebugStateManager | null = null;
    private state: DebugState = {
        enabled: false,
        logFullJson: false,
        enableDataDebug: false,
        enableWebsocketDebug: false,
        logBinaryHeaders: false
    };

    private constructor() {}

    public static getInstance(): DebugStateManager {
        if (!DebugStateManager.instance) {
            DebugStateManager.instance = new DebugStateManager();
        }
        return DebugStateManager.instance;
    }

    public async initialize(): Promise<void> {
        const settingsStore = SettingsStore.getInstance();
        await settingsStore.initialize();

        // Load initial debug settings
        this.state = {
            enabled: settingsStore.get('system.debug.enabled') as boolean ?? false,
            logFullJson: settingsStore.get('system.debug.log_full_json') as boolean ?? false,
            enableDataDebug: settingsStore.get('system.debug.enable_data_debug') as boolean ?? false,
            enableWebsocketDebug: settingsStore.get('system.debug.enable_websocket_debug') as boolean ?? false,
            logBinaryHeaders: settingsStore.get('system.debug.log_binary_headers') as boolean ?? false
        };

        // Subscribe to debug setting changes
        settingsStore.subscribe('system.debug.enabled', (_, value) => {
            this.state.enabled = value as boolean;
            this.updateLoggerConfig();
        });

        settingsStore.subscribe('system.debug.log_full_json', (_, value) => {
            this.state.logFullJson = value as boolean;
            this.updateLoggerConfig();
        });

        settingsStore.subscribe('system.debug.enable_data_debug', (_, value) => {
            this.state.enableDataDebug = value as boolean;
        });

        settingsStore.subscribe('system.debug.enable_websocket_debug', (_, value) => {
            this.state.enableWebsocketDebug = value as boolean;
        });

        settingsStore.subscribe('system.debug.log_binary_headers', (_, value) => {
            this.state.logBinaryHeaders = value as boolean;
        });

        this.updateLoggerConfig();
    }

    private updateLoggerConfig(): void {
        const { LoggerConfig } = require('./logger');
        LoggerConfig.setGlobalDebug(this.state.enabled);
        LoggerConfig.setFullJson(this.state.logFullJson);
    }

    public isEnabled(): boolean {
        return this.state.enabled;
    }

    public isWebsocketDebugEnabled(): boolean {
        return this.state.enabled && this.state.enableWebsocketDebug;
    }

    public isDataDebugEnabled(): boolean {
        return this.state.enabled && this.state.enableDataDebug;
    }

    public shouldLogBinaryHeaders(): boolean {
        return this.state.enabled && this.state.logBinaryHeaders;
    }

    public getState(): DebugState {
        return { ...this.state };
    }
}

export const debugState = DebugStateManager.getInstance();