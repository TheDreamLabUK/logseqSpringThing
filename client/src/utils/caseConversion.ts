/**
 * Utility functions for converting between different case styles
 */

/**
 * Converts a snake_case string to camelCase
 * @param str The snake_case string to convert
 * @returns The camelCase version of the string
 */
export function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Recursively converts all keys in an object from snake_case to camelCase
 * @param obj The object with snake_case keys
 * @returns A new object with all keys converted to camelCase
 */
export function convertSnakeToCamelCase<T extends Record<string, any>>(obj: T): Record<string, any> {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => convertSnakeToCamelCase(item)) as any;
  }

  return Object.keys(obj).reduce((result, key) => {
    const camelKey = snakeToCamel(key);
    const value = obj[key];
    
    result[camelKey] = typeof value === 'object' && value !== null
      ? convertSnakeToCamelCase(value)
      : value;
      
    return result;
  }, {} as Record<string, any>);
}
