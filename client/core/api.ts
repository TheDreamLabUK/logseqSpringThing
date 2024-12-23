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

// Helper function to build settings URL
export function buildSettingsUrl(category: string, setting?: string): string {
    const base = `settings/${category}`;
    return buildApiUrl(setting ? `${base}/${setting}` : base);
}

// Helper function to build WebSocket URL
export function buildWsUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/wss`;
}
