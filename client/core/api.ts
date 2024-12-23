/**
 * API configuration and utilities
 */

// Use relative URLs in both development and production
export const API_BASE = '';  // Empty string means use relative URLs

// Helper function to build API URLs
export function buildApiUrl(path: string): string {
    // Ensure path starts with /api/
    if (!path.startsWith('/api/')) {
        path = `/api/${path}`;
    }
    return `${API_BASE}${path}`;
}

// Helper function to build WebSocket URLs
export function buildWsUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/wss`;
}
