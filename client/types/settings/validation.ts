import { Settings } from './base';

export interface ValidationError {
    path: string;
    message: string;
    value?: any;
}

export interface ValidationResult {
    isValid: boolean;
    errors: ValidationError[];
}

interface ValidationRule {
    validate: (value: any) => boolean;
    message: string;
}

const validationRules: Record<string, Record<string, ValidationRule>> = {
    visualization: {
        'nodes.baseSize': {
            validate: (value: number) => value >= 0.1 && value <= 10,
            message: 'Base size must be between 0.1 and 10'
        },
        'nodes.opacity': {
            validate: (value: number) => value >= 0 && value <= 1,
            message: 'Opacity must be between 0 and 1'
        },
        'nodes.metalness': {
            validate: (value: number) => value >= 0 && value <= 1,
            message: 'Metalness must be between 0 and 1'
        },
        'nodes.roughness': {
            validate: (value: number) => value >= 0 && value <= 1,
            message: 'Roughness must be between 0 and 1'
        },
        'edges.width': {
            validate: (value: number) => value >= 0.1 && value <= 5,
            message: 'Edge width must be between 0.1 and 5'
        },
        'physics.attractionStrength': {
            validate: (value: number) => value >= 0 && value <= 2,
            message: 'Attraction strength must be between 0 and 2'
        },
        'physics.repulsionStrength': {
            validate: (value: number) => value >= 0 && value <= 2,
            message: 'Repulsion strength must be between 0 and 2'
        },
        'physics.springStrength': {
            validate: (value: number) => value >= 0 && value <= 2,
            message: 'Spring strength must be between 0 and 2'
        },
        'rendering.quality': {
            validate: (value: string) => ['low', 'medium', 'high'].includes(value),
            message: 'Quality must be low, medium, or high'
        },
        'rendering.ambientLightIntensity': {
            validate: (value: number) => value >= 0 && value <= 2,
            message: 'Ambient light intensity must be between 0 and 2'
        }
    },
    'visualization.bloom': {
        'visualization.bloom.strength': {
            validate: (value: number) => value >= 0 && value <= 5,
            message: 'Bloom strength must be between 0 and 5'
        },
        'visualization.bloom.radius': {
            validate: (value: number) => value >= 0 && value <= 3,
            message: 'Bloom radius must be between 0 and 3'
        },
        'visualization.bloom.edge_bloom_strength': {
            validate: (value: number) => value >= 0 && value <= 5,
            message: 'Edge bloom strength must be between 0 and 5'
        },
        'visualization.bloom.node_bloom_strength': {
            validate: (value: number) => value >= 0 && value <= 5,
            message: 'Node bloom strength must be between 0 and 5'
        },
        'visualization.bloom.environment_bloom_strength': {
            validate: (value: number) => value >= 0 && value <= 5,
            message: 'Environment bloom strength must be between 0 and 5'
        }
    }
};

export function validateSettings(settings: Partial<Settings>): ValidationResult {
    const errors: ValidationError[] = [];
    
    // Recursively validate all settings
    function validateObject(obj: any, path: string = '') {
        if (!obj || typeof obj !== 'object') return;
        
        Object.entries(obj).forEach(([key, value]) => {
            const currentPath = path ? `${path}.${key}` : key;
            
            // Check if there's a validation rule for this path
            for (const [category, rules] of Object.entries(validationRules)) {
                if (currentPath.startsWith(category)) {
                    const rule = rules[currentPath];
                    if (rule && !rule.validate(value)) {
                        errors.push({
                            path: currentPath,
                            message: rule.message,
                            value
                        });
                    }
                }
            }
            
            // Recursively validate nested objects
            if (value && typeof value === 'object' && !Array.isArray(value)) {
                validateObject(value, currentPath);
            }
        });
    }
    
    validateObject(settings);
    
    return {
        isValid: errors.length === 0,
        errors
    };
}

export function validateSettingValue(path: string, value: any, currentSettings: Settings): ValidationError[] {
    const errors: ValidationError[] = [];
    
    // Find matching validation rule
    for (const [category, rules] of Object.entries(validationRules)) {
        if (path.startsWith(category)) {
            const rule = rules[path];
            if (rule && !rule.validate(value)) {
                errors.push({
                    path,
                    message: rule.message,
                    value
                });
            }
        }
    }
    
    // Special validation for interdependent settings
    if (path.includes('physics')) {
        validatePhysicsSettings(path, value, currentSettings, errors);
    } else if (path.includes('rendering')) {
        validateRenderingSettings(path, value, currentSettings, errors);
    }
    
    return errors;
}

function validatePhysicsSettings(
    path: string,
    value: any,
    settings: Settings,
    errors: ValidationError[]
): void {
    const physics = settings.visualization.physics;
    
    // Example: Ensure attraction and repulsion strengths are balanced
    if (path === 'visualization.physics.attractionStrength' && physics.repulsionStrength) {
        const ratio = value / physics.repulsionStrength;
        if (ratio > 3 || ratio < 0.33) {
            errors.push({
                path,
                message: 'Attraction and repulsion strengths should be relatively balanced',
                value
            });
        }
    }
}

function validateRenderingSettings(
    path: string,
    value: any,
    settings: Settings,
    errors: ValidationError[]
): void {
    const rendering = settings.visualization.rendering;
    
    // Example: Warn about performance impact of combined settings
    if (path === 'visualization.rendering.quality' && value === 'high') {
        if (rendering.enableShadows && rendering.enableAmbientOcclusion) {
            errors.push({
                path,
                message: 'High quality with shadows and ambient occlusion may impact performance',
                value
            });
        }
    }
}

export function getValidationTooltip(path: string): string | undefined {
    for (const [category, rules] of Object.entries(validationRules)) {
        if (path.startsWith(category)) {
            const rule = rules[path];
            if (rule) {
                return rule.message;
            }
        }
    }
    return undefined;
}