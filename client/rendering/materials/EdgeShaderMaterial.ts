import { 
    ShaderMaterial, 
    Color, 
    Vector3, 
    WebGLRenderer, 
    MeshBasicMaterial
} from 'three';

import { Settings } from '../../types/settings/base';
import { createLogger } from '../../core/logger';
import { debugState } from '../../core/debugState';

const logger = createLogger('EdgeShaderMaterial');

// Three.js constants
const FRONT_SIDE = 0;  // THREE.FrontSide
const DOUBLE_SIDE = 2; // THREE.DoubleSide
const NORMAL_BLENDING = 1;  // THREE.NormalBlending
const ADDITIVE_BLENDING = 2;  // THREE.AdditiveBlending

export interface EdgeUniforms {
    [key: string]: { value: any };
    time: { value: number };
    opacity: { value: number };
    color: { value: Color };
    flowSpeed: { value: number };
    flowIntensity: { value: number };
    glowStrength: { value: number };
    distanceIntensity: { value: number };
    useGradient: { value: boolean };
    gradientColorA: { value: Color };
    gradientColorB: { value: Color };
    sourcePosition: { value: Vector3 };
    targetPosition: { value: Vector3 };
}

export class EdgeShaderMaterial extends ShaderMaterial {
    declare uniforms: EdgeUniforms;
    private updateFrequency: number;
    private frameCount: number = 0;
    private static instances: Set<EdgeShaderMaterial> = new Set();
    private static renderer: WebGLRenderer | null = null;
    private fallbackMaterial: MeshBasicMaterial | null = null;

    private validateUniforms(): boolean {
        if (!debugState.isShaderDebugEnabled()) return true;

        const uniformErrors: string[] = [];
        
        Object.entries(this.uniforms).forEach(([name, uniform]) => {
            if (uniform.value === undefined || uniform.value === null) {
                uniformErrors.push(`Uniform '${name}' has no value`);
            } else if (uniform.value instanceof Vector3 && 
                      (isNaN(uniform.value.x) || isNaN(uniform.value.y) || isNaN(uniform.value.z))) {
                uniformErrors.push(`Uniform '${name}' has invalid Vector3 value`);
            } else if (uniform.value instanceof Color) {
                const [r, g, b] = uniform.value.toArray();
                if (isNaN(r) || isNaN(g) || isNaN(b)) {
                    uniformErrors.push(`Uniform '${name}' has invalid Color value: ${(uniform.value as any).getHexString()}`);
                }
            } else if (typeof uniform.value === 'number' && 
                      (isNaN(uniform.value) || !isFinite(uniform.value))) {
                uniformErrors.push(`Uniform '${name}' has invalid number value`);
            }
        });

        if (uniformErrors.length > 0) {
            logger.shader('Uniform validation failed', { errors: uniformErrors });
            return false;
        }

        return true;
    }

    private checkWebGLVersion(renderer: WebGLRenderer): boolean {
        const gl = renderer.domElement.getContext('webgl2') || renderer.domElement.getContext('webgl');
        const isWebGL2 = gl instanceof WebGL2RenderingContext;
        
        if (debugState.isShaderDebugEnabled()) {
            logger.shader('WebGL context check', {
                version: isWebGL2 ? '2.0' : '1.0',
                glslVersion: (this as any).glslVersion ?? 'none'
            });
        }
        return isWebGL2;
    }

    constructor(settings: Settings, context: 'ar' | 'desktop' = 'desktop') {
        const isAR = context === 'ar';

        if (debugState.isShaderDebugEnabled()) {
            logger.shader('Creating EdgeShaderMaterial', { context, settings });
        }
        
        super({
            uniforms: {
                time: { value: 0 },
                opacity: { value: settings.visualization.edges.opacity },
                color: { value: new Color(settings.visualization.edges.color) },
                flowSpeed: { value: settings.visualization.edges.flowSpeed },
                flowIntensity: { value: settings.visualization.edges.flowIntensity },
                glowStrength: { value: settings.visualization.edges.glowStrength },
                distanceIntensity: { value: settings.visualization.edges.distanceIntensity },
                useGradient: { value: settings.visualization.edges.useGradient },
                gradientColorA: { value: new Color(settings.visualization.edges.gradientColors[0]) },
                gradientColorB: { value: new Color(settings.visualization.edges.gradientColors[1]) },
                sourcePosition: { value: new Vector3() },
                targetPosition: { value: new Vector3() }
            },
            vertexShader: `
                varying vec2 vUv;
                varying vec3 vPosition;
                varying float vDistance;
                const float PI = 3.14159265359;
                
                uniform vec3 sourcePosition;
                uniform vec3 targetPosition;
                
                void main() {
                    vUv = uv;
                    vPosition = position;
                    
                    // Optimize distance calculation
                    vec3 edgeDir = normalize(targetPosition - sourcePosition);
                    vec3 posVector = position - sourcePosition;
                    vDistance = dot(edgeDir, normalize(posVector));
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float opacity;
                uniform vec3 color;
                uniform float flowSpeed;
                uniform float flowIntensity;
                uniform float glowStrength;
                uniform float distanceIntensity;
                uniform bool useGradient;
                uniform vec3 gradientColorA;
                uniform vec3 gradientColorB;
                
                varying vec2 vUv;
                varying vec3 vPosition;
                varying float vDistance;
                
                void main() {
                    // Simplified flow calculation
                    float flow = sin(vDistance * 8.0 - time * flowSpeed) * 0.5 + 0.5;
                    flow *= flowIntensity;

                    // Optimized distance-based intensity
                    float distanceFactor = 1.0 - abs(vDistance - 0.5) * 2.0;
                    distanceFactor = pow(distanceFactor, distanceIntensity);
                    
                    // Base color with gradient
                    vec3 finalColor = useGradient ? 
                        mix(gradientColorA, gradientColorB, vDistance) : 
                        color;

                    // Add flow and glow effects
                    finalColor += flow * 0.2;
                    finalColor += (1.0 - vUv.y) * glowStrength * 0.3;
                    
                    // Apply distance factor
                    finalColor *= mix(0.5, 1.0, distanceFactor);
                    
                    gl_FragColor = vec4(finalColor, opacity * (0.7 + flow * 0.3));
                }
            `,
            transparent: true,
            side: isAR ? FRONT_SIDE : DOUBLE_SIDE,
            blending: isAR ? NORMAL_BLENDING : ADDITIVE_BLENDING,
            depthWrite: !isAR
        });

        // Set update frequency based on context
        this.updateFrequency = isAR ? 3 : 2; // Update less frequently in AR

        // Add this instance to the set of instances
        EdgeShaderMaterial.instances.add(this);

        if (EdgeShaderMaterial.renderer) {
            try {
                if (debugState.isShaderDebugEnabled()) {
                    logger.shader('Attempting shader compilation', {
                        context,
                        hasRenderer: true,
                        webgl2: this.checkWebGLVersion(EdgeShaderMaterial.renderer)
                    });
                }

                // Validate uniforms before compilation
                if (!this.validateUniforms()) {
                    throw new Error('Uniform validation failed');
                }

                this.needsUpdate = true;
            } catch (error) {
                const err = error instanceof Error ? error : new Error('Shader compilation failed');
                if (debugState.isShaderDebugEnabled()) {
                    logger.shader('Shader compilation failed', {
                        error: err,
                        message: err.message,
                        stack: err.stack,
                        context
                    });
                }

                this.fallbackMaterial = new MeshBasicMaterial({
                    color: settings.visualization.edges.color,
                    wireframe: true,
                    transparent: true,
                    opacity: settings.visualization.edges.opacity
                });
            }
        }
    }

    public static setRenderer(renderer: WebGLRenderer): void {
        EdgeShaderMaterial.renderer = renderer;
        
        if (debugState.isShaderDebugEnabled()) {
            const gl = renderer.domElement.getContext('webgl2') || renderer.domElement.getContext('webgl');
            if (gl) {
                logger.shader('Renderer initialized', {
                    isWebGL2: gl instanceof WebGL2RenderingContext,
                    maxTextures: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
                    maxVaryings: gl.getParameter(gl.MAX_VARYING_VECTORS),
                    maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
                    maxVertexUniforms: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
                    maxFragmentUniforms: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS)
                });
            }
        }

        // Force shader compilation for all instances
        EdgeShaderMaterial.instances.forEach(instance => {
            try {
                if (!instance.validateUniforms()) {
                    logger.shader('Skipping recompilation due to invalid uniforms');
                    return;
                }

                instance.needsUpdate = true;
                
                // If we had a fallback material and compilation succeeded, remove it
                if (instance.fallbackMaterial) {
                    instance.fallbackMaterial.dispose();
                    instance.fallbackMaterial = null;
                }
            } catch (error) {
                const err = error instanceof Error ? error : new Error('Shader recompilation failed');
                if (debugState.isShaderDebugEnabled()) {
                    logger.shader('Shader recompilation failed', {
                        error: err,
                        message: err.message,
                        stack: err.stack
                    });
                }

                if (!instance.fallbackMaterial) {
                    instance.fallbackMaterial = new MeshBasicMaterial({
                        color: instance.uniforms.color.value,
                        wireframe: true,
                        transparent: true,
                        opacity: instance.uniforms.opacity.value
                    });
                }
            }
        });
    }

    update(deltaTime: number): void {
        this.frameCount++;
        if (this.frameCount % this.updateFrequency === 0) {
            const oldTime = this.uniforms.time.value;
            this.uniforms.time.value += deltaTime;

            if (debugState.isShaderDebugEnabled() && 
                (isNaN(this.uniforms.time.value) || !isFinite(this.uniforms.time.value))) {
                logger.shader('Invalid time value detected', {
                    oldTime,
                    deltaTime,
                    newTime: this.uniforms.time.value
                });
                this.uniforms.time.value = 0;
            }
        }
    }

    setSourceTarget(source: Vector3, target: Vector3): void {
        if (debugState.isShaderDebugEnabled()) {
            logger.shader('Setting source/target positions', {
                source,
                target
            });
        }

        this.uniforms.sourcePosition.value.copy(source);
        this.uniforms.targetPosition.value.copy(target);
    }

    clone(): this {
        if (debugState.isShaderDebugEnabled()) {
            logger.shader('Cloning EdgeShaderMaterial');
        }

        const material = new EdgeShaderMaterial({
            visualization: {
                edges: {
                    opacity: this.uniforms.opacity.value,
                    color: (this.uniforms.color.value as any).getHexString(),
                    flowSpeed: this.uniforms.flowSpeed.value,
                    flowIntensity: this.uniforms.flowIntensity.value,
                    glowStrength: this.uniforms.glowStrength.value,
                    distanceIntensity: this.uniforms.distanceIntensity.value,
                    useGradient: this.uniforms.useGradient.value,
                    gradientColors: [
                        (this.uniforms.gradientColorA.value as any).getHexString(),
                        (this.uniforms.gradientColorB.value as any).getHexString()
                    ]
                }
            }
        } as Settings);

        if (debugState.isShaderDebugEnabled()) {
            logger.shader('Material cloned successfully');
        }

        return material as this;
    }

    dispose(): void {
        // Remove this instance from the set when disposed
        EdgeShaderMaterial.instances.delete(this);
        // Dispose of fallback material if it exists
        if (this.fallbackMaterial) {
            this.fallbackMaterial.dispose();
        }
        // Call parent dispose
        super.dispose();

        if (debugState.isShaderDebugEnabled()) {
            logger.shader('Material disposed');
        }
    }
}