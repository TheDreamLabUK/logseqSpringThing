import { API_BASE, API_ENDPOINTS, SETTINGS_CATEGORIES } from './constants';

// Helper function to build API URLs
export function buildApiUrl(path: string): string {
    // Handle both endpoint constants and dynamic paths
    if (path.startsWith('/api/')) {
        return `${API_BASE}${path}`;
    }
    return `${API_BASE}/api/${path}`;
}

// Helper function to build settings URL
export function buildSettingsUrl(category: keyof typeof SETTINGS_CATEGORIES, setting?: string): string {
    // Get snake_case category from enum
    const categorySnake = SETTINGS_CATEGORIES[category];
    if (!categorySnake) {
        throw new Error(`Invalid settings category: ${category}`);
    }
    
    // Convert setting to snake_case if provided
    const settingSnake = setting?.replace(/-/g, '_');
    
    const base = settingSnake 
        ? `${API_ENDPOINTS.SETTINGS}/${categorySnake}/${settingSnake}`
        : `${API_ENDPOINTS.SETTINGS}/${categorySnake}`;
    return base;
}

// Helper function to build graph URL
export function buildGraphUrl(type: 'data' | 'update' | 'paginated', params?: Record<string, string>): string {
    let endpoint: string;
    switch (type) {
        case 'data':
            endpoint = API_ENDPOINTS.GRAPH_DATA;
            break;
        case 'update':
            endpoint = API_ENDPOINTS.GRAPH_UPDATE;
            break;
        case 'paginated':
            endpoint = API_ENDPOINTS.GRAPH_PAGINATED;
            break;
        default:
            throw new Error(`Invalid graph endpoint type: ${type}`);
    }
    
    if (!params) return endpoint;
    
    const queryString = Object.entries(params)
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
        .join('&');
    return `${endpoint}?${queryString}`;
}

// Helper function to build files URL
export function buildFilesUrl(path: string): string {
    return `${API_ENDPOINTS.FILES}/${path}`;
}

// Helper function to build WebSocket URL
export function buildWsUrl(): string {
    const isProduction = ['www.visionflow.info', 'visionflow.info'].includes(window.location.hostname);
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    
    if (isProduction) {
        // In production, always use wss:// with the domain
        return `wss://www.visionflow.info/wss`;
    } else {
        // In development, use the current host with ws:// or wss://
        return `${protocol}//${host}/wss`;
    }
}
