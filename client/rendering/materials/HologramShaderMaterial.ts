import { 
    ShaderMaterial, 
    Color, 
    Vector3, 
    AdditiveBlending, 
    WebGLRenderer, 
    MeshBasicMaterial
} from 'three';
import { createLogger } from '../../core/logger';
import { debugState } from '../../core/debugState';

export interface HologramUniforms {
    [key: string]: { value: any };
    time: { value: number };
    opacity: { value: number };
    color: { value: Color };
    pulseIntensity: { value: number };
    interactionPoint: { value: Vector3 };
    interactionStrength: { value: number };
    isEdgeOnly: { value: boolean };
}

const logger = createLogger('HologramShaderMaterial');

// Three.js side constants
const FRONT_SIDE = 0;  // THREE.FrontSide
const DOUBLE_SIDE = 2; // THREE.DoubleSide

export class HologramShaderMaterial extends ShaderMaterial {
    declare uniforms: HologramUniforms;
    private static renderer: WebGLRenderer | null = null;
    private static instances: Set<HologramShaderMaterial> = new Set();
    private updateFrequency: number;
    private frameCount: number;
    private fallbackMaterial: MeshBasicMaterial | null = null;
    public wireframe = false;

    private validateUniforms(): boolean {
        if (!debugState.isShaderDebugEnabled()) return true;

        const uniformErrors: string[] = [];
        
        // Check each uniform
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

    constructor(settings?: any, context: 'ar' | 'desktop' = 'desktop') {
        if (debugState.isDataDebugEnabled()) {
            logger.debug('Creating HologramShaderMaterial', { context, settings });
        }
        const isAR = context === 'ar';
        
        // Check WebGL version to determine which shader version to use
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        const isWebGL2 = !!gl;
        
        // Choose appropriate vertex shader based on WebGL version
        const vertexShader = isWebGL2 ? 
            /* WebGL2 vertex shader */
            `#version 300 es
            in vec2 uv;
            in vec3 normal;
            in vec3 position;
            
            uniform mat4 modelMatrix;
            uniform mat4 viewMatrix;
            uniform mat4 projectionMatrix;
            uniform float time;
            uniform vec3 interactionPoint;
            uniform float interactionStrength;
            
            out vec2 vUv;
            out vec3 vNormal;
            out vec3 vPosition;
            
            void main() {
                vUv = uv;
                vNormal = normalize(normalMatrix * normal);
                
                // Apply model matrix to get world position
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vPosition = worldPosition.xyz;
                
                // Calculate distance to interaction point
                float dist = distance(worldPosition.xyz, interactionPoint);
                float influence = max(0.0, 1.0 - dist / 2.0) * interactionStrength;
                
                // Apply view and projection matrices
                gl_Position = projectionMatrix * viewMatrix * worldPosition;
            }` :
            /* WebGL1 vertex shader */
            `attribute vec2 uv;
            attribute vec3 normal;
            attribute vec3 position;
            
            uniform mat4 modelMatrix;
            uniform mat4 viewMatrix;
            uniform mat4 projectionMatrix;
            uniform mat3 normalMatrix;
            uniform float time;
            uniform vec3 interactionPoint;
            uniform float interactionStrength;
            
            varying vec2 vUv;
            varying vec3 vNormal;
            varying vec3 vPosition;
            
            void main() {
                vUv = uv;
                vNormal = normalize(normalMatrix * normal);
                
                // Apply model matrix to get world position
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vPosition = worldPosition.xyz;
                
                // Calculate distance to interaction point
                float dist = distance(worldPosition.xyz, interactionPoint);
                float influence = max(0.0, 1.0 - dist / 2.0) * interactionStrength;
                
                // Apply view and projection matrices
                gl_Position = projectionMatrix * viewMatrix * worldPosition;
            }`;
            
        // Choose appropriate fragment shader based on WebGL version
        const fragmentShader = isWebGL2 ?
            /* WebGL2 fragment shader */
            `#version 300 es
            precision highp float;
            
            uniform float time;
            uniform float opacity;
            uniform vec3 color;
            uniform float pulseIntensity;
            uniform bool isEdgeOnly;
            
            in vec2 vUv;
            in vec3 vNormal;
            in vec3 vPosition;
            
            out vec4 fragColor;
            
            void main() {
                // Edge detection based on normal
                float edgeFactor = abs(dot(normalize(vNormal), normalize(vec3(0.0, 0.0, 1.0))));
                edgeFactor = 1.0 - pow(edgeFactor, 2.0);
                
                // Pulse effect
                float pulse = sin(time * 2.0) * 0.5 + 0.5;
                pulse = pulse * pulseIntensity;
                
                // Grid pattern
                float gridSize = 20.0;
                vec2 grid = fract(vUv * gridSize);
                float gridLine = step(0.95, grid.x) + step(0.95, grid.y);
                
                // Combine effects
                float finalOpacity = opacity;
                vec3 finalColor = color;
                
                if (isEdgeOnly) {
                    // Edge-only mode
                    finalOpacity = edgeFactor * opacity * (1.0 + pulse * 0.3);
                    finalColor = mix(color, color * 1.5, pulse);
                } else {
                    // Full hologram mode
                    finalOpacity = mix(0.1, opacity, edgeFactor) * (1.0 + pulse * 0.3);
                    finalColor = mix(color * 0.5, color * 1.2, edgeFactor);
                    
                    // Add grid lines
                    finalOpacity = mix(finalOpacity, opacity, gridLine * 0.7);
                    finalColor = mix(finalColor, color * 1.5, gridLine * 0.7);
                }
                
                fragColor = vec4(finalColor, finalOpacity);
            }` :
            /* WebGL1 fragment shader */
            `precision highp float;
            
            uniform float time;
            uniform float opacity;
            uniform vec3 color;
            uniform float pulseIntensity;
            uniform bool isEdgeOnly;
            
            varying vec2 vUv;
            varying vec3 vNormal;
            varying vec3 vPosition;
            
            void main() {
                // Edge detection based on normal
                float edgeFactor = abs(dot(normalize(vNormal), normalize(vec3(0.0, 0.0, 1.0))));
                edgeFactor = 1.0 - pow(edgeFactor, 2.0);
                
                // Pulse effect
                float pulse = sin(time * 2.0) * 0.5 + 0.5;
                pulse = pulse * pulseIntensity;
                
                // Grid pattern
                float gridSize = 20.0;
                vec2 grid = fract(vUv * gridSize);
                float gridLine = step(0.95, grid.x) + step(0.95, grid.y);
                
                // Combine effects
                float finalOpacity = opacity;
                vec3 finalColor = color;
                
                if (isEdgeOnly) {
                    // Edge-only mode
                    finalOpacity = edgeFactor * opacity * (1.0 + pulse * 0.3);
                    finalColor = mix(color, color * 1.5, pulse);
                } else {
                    // Full hologram mode
                    finalOpacity = mix(0.1, opacity, edgeFactor) * (1.0 + pulse * 0.3);
                    finalColor = mix(color * 0.5, color * 1.2, edgeFactor);
                    
                    // Add grid lines
                    finalOpacity = mix(finalOpacity, opacity, gridLine * 0.7);
                    finalColor = mix(finalColor, color * 1.5, gridLine * 0.7);
                }
                
                gl_FragColor = vec4(finalColor, finalOpacity);
            }`;
        
        super({
            uniforms: {
                time: { value: 0 },
                opacity: { value: settings?.visualization?.hologram?.opacity ?? 1.0 },
                color: { value: new Color(settings?.visualization?.hologram?.color ?? 0x00ff00) },
                pulseIntensity: { value: isAR ? 0.1 : 0.2 },
                interactionPoint: { value: new Vector3() },
                interactionStrength: { value: 0.0 },
                isEdgeOnly: { value: false }
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            blending: AdditiveBlending,
            side: DOUBLE_SIDE,
            depthWrite: false
        });

        this.updateFrequency = isAR ? 2 : 1; // Update every frame in desktop, every other frame in AR
        this.frameCount = 0;
        
        // Add this instance to the set of instances
        HologramShaderMaterial.instances.add(this);
        
        if (debugState.isDataDebugEnabled()) {
            logger.debug('HologramShaderMaterial initialized', { updateFrequency: this.updateFrequency });
        }

        // If renderer is already set, compile shader
        if (HologramShaderMaterial.renderer) {
            try {
                if (debugState.isShaderDebugEnabled()) {
                    logger.shader('Attempting shader compilation', {
                        context,
                        hasRenderer: true,
                        webgl2: this.checkWebGLVersion(HologramShaderMaterial.renderer)
                    });
                }

                // Validate uniforms before compilation
                if (!this.validateUniforms()) {
                    throw new Error('Uniform validation failed');
                }

                this.needsUpdate = true;

            } catch (error: unknown) {
                const errorMessage = error instanceof Error ? error : new Error(String(error));
                const errorStack = error instanceof Error ? error.stack : undefined;
                if (debugState.isShaderDebugEnabled()) {
                    logger.shader('Shader compilation failed', {
                        error: errorMessage,
                        stack: errorStack,
                        context
                    });
                }

                // Create fallback material
                this.fallbackMaterial = new MeshBasicMaterial({
                    color: settings?.visualization?.hologram?.color ?? 0x00ff00,
                    wireframe: true,
                    transparent: this.transparent,
                    opacity: settings?.visualization?.hologram?.opacity ?? 0.5
                });
                // Copy necessary properties
                this.needsUpdate = true;
                this.side = isAR ? FRONT_SIDE : DOUBLE_SIDE;
                this.transparent = true;
            }
        }
    }

    public static setRenderer(renderer: WebGLRenderer): void {
        HologramShaderMaterial.renderer = renderer;
        
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
        HologramShaderMaterial.instances.forEach(async instance => {
            try {
                if (!instance.validateUniforms()) {
                    logger.shader('Skipping recompilation due to invalid uniforms');
                    return;
                }

                instance.needsUpdate = true;
                instance.needsUpdate = true;
                
                // If we had a fallback material and compilation succeeded, remove it
                if (instance.fallbackMaterial) {
                    instance.fallbackMaterial.dispose();
                    instance.fallbackMaterial = null;
                }
            } catch (error: unknown) {
                const errorMessage = error instanceof Error ? error : new Error(String(error));
                const errorStack = error instanceof Error ? error.stack : undefined;
                if (debugState.isShaderDebugEnabled()) {
                    logger.shader('Shader recompilation failed', {
                        error: errorMessage,
                        stack: errorStack
                    });
                }
                // Create fallback material if we don't have one
                if (!instance.fallbackMaterial) {
                    instance.fallbackMaterial = new MeshBasicMaterial({
                        color: instance.uniforms.color.value,
                        wireframe: instance.wireframe,
                        transparent: true,
                        opacity: instance.uniforms.opacity.value,
                        side: instance.side
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

            if (this.uniforms.interactionStrength.value > 0.01) {
                this.uniforms.interactionStrength.value *= 0.95; // Decay interaction effect
                this.validateUniforms(); // Check uniforms after update
            }
        }
    }

    handleInteraction(position: Vector3): void {
        if (this.frameCount % this.updateFrequency === 0) {
            this.uniforms.interactionPoint.value.copy(position);
            this.uniforms.interactionStrength.value = 1.0;
        }
    }

    setEdgeOnly(enabled: boolean): void {
        this.uniforms.isEdgeOnly.value = enabled;
        // Increase pulse intensity for better visibility in wireframe mode
        this.uniforms.pulseIntensity.value = enabled ? (this.side === 0 ? 0.15 : 0.3) : (this.side === 0 ? 0.1 : 0.2);
    }

    clone(): this {
        if (debugState.isDataDebugEnabled()) {
            logger.debug('Cloning HologramShaderMaterial');
        }
        // Create settings object from current uniforms
        const settings = {
            visualization: {
                hologram: {
                    opacity: this.uniforms.opacity.value,
                    color: '#' + (this.uniforms.color.value as any).getHexString()
                }
            }
        };
        if (debugState.isDataDebugEnabled()) {
            logger.debug('Clone settings', settings);
        }
        const material = new HologramShaderMaterial(settings, this.side === 0 ? 'ar' : 'desktop');
        material.uniforms = {
            time: { value: this.uniforms.time.value },
            opacity: { value: this.uniforms.opacity.value },
            color: { value: this.uniforms.color.value.clone() },
            pulseIntensity: { value: this.uniforms.pulseIntensity.value },
            interactionPoint: { value: this.uniforms.interactionPoint.value.clone() },
            interactionStrength: { value: this.uniforms.interactionStrength.value },
            isEdgeOnly: { value: this.uniforms.isEdgeOnly.value }
        };

        if (debugState.isDataDebugEnabled()) {
            logger.debug('Material cloned successfully');
        }
        return material as this;
    }

    dispose(): void {
        // Remove this instance from the set when disposed
        HologramShaderMaterial.instances.delete(this);
        // Dispose of fallback material if it exists
        if (this.fallbackMaterial) {
            this.fallbackMaterial.dispose();
        }
        // Call parent dispose
        super.dispose();
    }
}
