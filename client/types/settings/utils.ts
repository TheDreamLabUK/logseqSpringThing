import { Settings } from '../settings';
import { defaultSettings } from '../../state/defaultSettings';

// Type for top-level settings categories
export type SettingsCategory = keyof Settings;

// Type for all possible paths in settings
export type SettingsPath = string;

// Type guard to check if a string is a valid settings category
export function isSettingsCategory(key: string): key is SettingsCategory {
    return key in defaultSettings;
}

// Type guard to check if a path exists in settings
export function isValidSettingPath(path: string): boolean {
    if (!path) {
        return false;
    }

    try {
        const parts = path.split('.');
        if (parts.length === 0) {
            return false;
        }

        let current: any = defaultSettings;
        for (const part of parts) {
            if (!part || typeof part !== 'string' || !(part in current)) {
                return false;
            }
            current = current[part];
        }
        
        return true;
    } catch (_error: unknown) {
        return false;
    }
}

// Get value from settings using path
export function getSettingValue(settings: Settings, path: string): any {
    if (!settings || typeof settings !== 'object') {
        throw new Error('Invalid settings object');
    }
    if (!path) {
        throw new Error('Path cannot be empty');
    }
    
    try {
        return path.split('.').reduce((obj: any, key) => {
            if (obj === null || obj === undefined) {
                throw new Error(`Invalid path: ${path}`);
            }
            return obj[key];
        }, settings);
    } catch (_error: unknown) {
        const message = 'Unknown error';
        throw new Error(`Failed to get setting value at path ${path}: ${message}`);
    }
}

// Set value in settings using path
export function setSettingValue(settings: Settings, path: string, value: any): void {
    if (!settings || typeof settings !== 'object') {
        throw new Error('Invalid settings object');
    }
    if (!path) {
        throw new Error('Path cannot be empty');
    }
    
    try {
        const parts = path.split('.');
        const lastKey = parts.pop();
        if (!lastKey) {
            throw new Error('Invalid path format');
        }
        
        const target = parts.reduce((obj: any, key) => {
            if (!(key in obj)) {
                obj[key] = {};
            }
            return obj[key];
        }, settings);

        if (!target || typeof target !== 'object') {
            throw new Error(`Invalid path: ${path}`);
        }

        target[lastKey] = value;
    } catch (_error: unknown) {
        const message = 'Unknown error';
        throw new Error(`Failed to set setting value at path ${path}: ${message}`);
    }
}

// Get the parent category of a setting path
export function getSettingCategory(path: string): SettingsCategory | undefined {
    if (!path) {
        return undefined;
    }
    const category = path.split('.')[0];
    return isSettingsCategory(category) ? category : undefined;
}

// Get subcategory path (everything after the main category)
export function getSettingSubPath(path: string): string | undefined {
    if (!path) {
        return undefined;
    }
    const parts = path.split('.');
    return parts.length > 1 ? parts.slice(1).join('.') : undefined;
}

// Helper to check if a value is a nested settings object
export function isSettingsObject(value: any): boolean {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// Helper to get all paths in a settings object
export function getAllSettingPaths(
    obj: any,
    parentPath: string = '',
    paths: string[] = []
): string[] {
    if (!isSettingsObject(obj)) {
        return paths;
    }

    for (const key in obj) {
        const currentPath = parentPath ? `${parentPath}.${key}` : key;
        if (isSettingsObject(obj[key])) {
            getAllSettingPaths(obj[key], currentPath, paths);
        } else {
            paths.push(currentPath);
        }
    }
    return paths;
}

// Type helper for settings values
export type SettingValue = string | number | boolean | string[] | number[];

// Helper to get the appropriate input type for a setting
export function getSettingInputType(value: SettingValue): string {
    if (value === null || value === undefined) {
        return 'text';
    }
    if (typeof value === 'boolean') return 'checkbox';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'string' && value.startsWith('#')) return 'color';
    if (Array.isArray(value)) return 'select';
    return 'text';
}

// Helper to format setting names for display
export function formatSettingName(setting: string): string {
    if (!setting) return '';
    return setting
        .split(/(?=[A-Z])|_/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// Helper to get step value for number inputs
export function getStepValue(key: string): string {
    if (!key) return '1';
    return key.toLowerCase().match(/strength|opacity|intensity|threshold|scale/)
        ? '0.1'
        : '1';
}
