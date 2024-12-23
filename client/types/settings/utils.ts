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
    const parts = path.split('.');
    let current: any = defaultSettings;
    
    for (const part of parts) {
        if (!(part in current)) {
            return false;
        }
        current = current[part];
    }
    
    return true;
}

// Get value from settings using path
export function getSettingValue(settings: Settings, path: string): any {
    return path.split('.').reduce((obj: any, key) => obj && obj[key], settings);
}

// Set value in settings using path
export function setSettingValue(settings: Settings, path: string, value: any): void {
    const parts = path.split('.');
    const lastKey = parts.pop()!;
    const target = parts.reduce((obj: any, key) => obj && obj[key], settings);
    if (target) {
        target[lastKey] = value;
    }
}

// Get the parent category of a setting path
export function getSettingCategory(path: string): SettingsCategory | undefined {
    const category = path.split('.')[0];
    return isSettingsCategory(category) ? category : undefined;
}

// Get subcategory path (everything after the main category)
export function getSettingSubPath(path: string): string | undefined {
    const parts = path.split('.');
    return parts.length > 1 ? parts.slice(1).join('.') : undefined;
}

// Helper to check if a value is a nested settings object
export function isSettingsObject(value: any): boolean {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// Helper to get all paths in a settings object
export function getAllSettingPaths(
    obj: any,
    parentPath: string = '',
    paths: string[] = []
): string[] {
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
    if (typeof value === 'boolean') return 'checkbox';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'string' && value.startsWith('#')) return 'color';
    return 'text';
}

// Helper to format setting names for display
export function formatSettingName(setting: string): string {
    return setting
        .split(/(?=[A-Z])|_/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// Helper to get step value for number inputs
export function getStepValue(key: string): string {
    return key.toLowerCase().match(/strength|opacity|intensity|threshold|scale/)
        ? '0.1'
        : '1';
}
