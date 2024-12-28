// Cache for converted strings
const conversionCache = new Map<string, string>();

export function toKebabCase(str: string): string {
    const cached = conversionCache.get(str);
    if (cached) {
        return cached;
    }

    const result = str
        .replace(/([a-z])([A-Z])/g, '$1-$2')
        .replace(/[\s_]+/g, '-')
        .toLowerCase();

    conversionCache.set(str, result);
    return result;
}

export function toCamelCase(str: string): string {
    const cached = conversionCache.get(str);
    if (cached) {
        return cached;
    }

    const result = str
        .replace(/[-_\s]+(.)?/g, (_, c) => c ? c.toUpperCase() : '')
        .replace(/^(.)/, c => c.toLowerCase());

    conversionCache.set(str, result);
    return result;
}

// Clear cache if it gets too large
export function clearConversionCache(): void {
    if (conversionCache.size > 1000) {
        conversionCache.clear();
    }
} 