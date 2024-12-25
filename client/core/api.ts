/**
 * API configuration and utilities
 */

import { API_BASE, API_PATHS, API_ENDPOINTS } from './constants';

// Helper function to build API URLs
export function buildApiUrl(path: string): string {
    // Handle both endpoint constants and dynamic paths
    if (path.startsWith('/api/')) {
        return `${API_BASE}${path}`;
    }
    return `${API_BASE}/api/${path}`;
}

// Helper function to build settings URL
export function buildSettingsUrl(category: string, setting?: string): string {
    const base = setting 
        ? `${API_ENDPOINTS.SETTINGS_BASE}/${category}/${setting}`
        : `${API_ENDPOINTS.SETTINGS_BASE}/${category}`;
    return base;
}

// Helper function to build visualization URL
export function buildVisualizationUrl(path: string): string {
    return `${API_ENDPOINTS.VISUALIZATION}/${path}`;
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
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/${API_PATHS.WEBSOCKET}`;
}
