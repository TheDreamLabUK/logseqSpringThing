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
        'physics.iterations': {
            validate: (value: number) => value >= 1 && value <= 500,
            message: 'Iterations must be between 1 and 500'
        },
        'physics.springStrength': {
            validate: (value: number) => value >= 0.1 && value <= 10,
            message: 'Spring strength must be between 0.1 and 10'
        },
        'physics.repulsionStrength': {
            validate: (value: number) => value >= 1 && value <= 2000,
            message: 'Repulsion strength must be between 1 and 2000'
        },
        'physics.repulsionDistance': {
            validate: (value: number) => value >= 100 && value <= 2000,
            message: 'Repulsion distance must be between 100 and 2000'
        },
        'physics.massScale': {
            validate: (value: number) => value >= 0.1 && value <= 5,
            message: 'Mass scale must be between 0.1 and 5'
        },
        'physics.damping': {
            validate: (value: number) => value >= 0 && value <= 1,
            message: 'Damping must be between 0 and 1'
        },
        'physics.boundaryDamping': {
            validate: (value: number) => value >= 0.5 && value <= 1,
            message: 'Boundary damping must be between 0.5 and 1'
        },
        'physics.boundsSize': {
            validate: (value: number) => value >= 100 && value <= 5000,
            message: 'Bounds size must be between 100 and 5000'
        },
        'physics.enableBounds': {
            validate: (value: boolean) => typeof value === 'boolean',
            message: 'Enable bounds must be a boolean'
        },
        'physics.collisionRadius': {
            validate: (value: number) => value >= 0.1 && value <= 100,
            message: 'Collision radius must be between 0.1 and 100'
        },
        'physics.maxVelocity': {
            validate: (value: number) => value >= 0.1 && value <= 10,
            message: 'Max velocity must be between 0.1 and 10'
        },
        'physics.attractionStrength': {
            validate: (value: number) => value >= 0.1 && value <= 10,
            message: 'Attraction strength must be between 0.1 and 10'
        }
    }
};

export function validateSettings(settings: Partial<Settings>): ValidationResult {
    const errors: ValidationError[] = [];
    
    function validateObject(obj: any, path: string = '') {
        if (!obj || typeof obj !== 'object') return;
        
        Object.entries(obj).forEach(([key, value]) => {
            const currentPath = path ? `${path}.${key}` : key;
            
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
    
    if (path.includes('physics')) {
        validatePhysicsSettings(path, value, currentSettings, errors);
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
    
    // Validate repulsion and distance relationship
    if (path === 'visualization.physics.repulsionStrength' && physics.repulsionDistance) {
        if (value > physics.repulsionDistance) {
            errors.push({
                path,
                message: 'Repulsion strength should not exceed repulsion distance for stability',
                value
            });
        }
    }

    // Validate performance impact of iterations
    if (path === 'visualization.physics.iterations' && value > 200) {
        errors.push({
            path,
            message: 'High iteration count may impact performance',
            value
        });
    }

    // Validate mass scale impact on forces
    if (path === 'visualization.physics.massScale' && value > 2) {
        errors.push({
            path,
            message: 'High mass scale values may cause unstable behavior',
            value
        });
    }

    // Validate bounds size and repulsion distance relationship
    if (path === 'visualization.physics.boundsSize' && physics.repulsionDistance) {
        if (value < physics.repulsionDistance * 0.5) {
            errors.push({
                path,
                message: 'Bounds size should be at least half of repulsion distance',
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