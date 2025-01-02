export const debugConfig = {
    // Enable/disable debug mode globally
    enabled: true,

    // Configure specific component debugging
    components: {
        webxr: {
            enabled: true,
            logLevel: 'debug',
            performance: true,
        },
        graph: {
            enabled: true,
            logLevel: 'debug',
            performance: true,
        },
        websocket: {
            enabled: true,
            logLevel: 'debug',
            connectionEvents: true,
            messageEvents: true,
        },
        ragflow: {
            enabled: true,
            logLevel: 'debug',
            requests: true,
            responses: true,
        },
        perplexity: {
            enabled: true,
            logLevel: 'debug',
            queries: true,
            results: true,
        },
        audio: {
            enabled: true,
            logLevel: 'debug',
            synthesis: true,
            streaming: true,
        }
    },

    // Configure performance monitoring
    performance: {
        enabled: true,
        frameRate: true,
        memory: true,
        networkLatency: true,
    },

    // Configure console output
    console: {
        useColors: true,
        showTimestamp: true,
        showNamespace: true,
        groupRelatedLogs: true,
    }
};
