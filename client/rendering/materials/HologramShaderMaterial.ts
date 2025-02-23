import { ShaderMaterial, Color, Vector3, AdditiveBlending, WebGLRenderer } from 'three';
import { createLogger } from '../../core/logger';

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

export class HologramShaderMaterial extends ShaderMaterial {
    declare uniforms: HologramUniforms;
    private static renderer: WebGLRenderer | null = null;
    private updateFrequency: number;
    private frameCount: number;

    constructor(settings?: any, context: 'ar' | 'desktop' = 'desktop') {
        logger.debug('Creating HologramShaderMaterial', { context, settings });
        const isAR = context === 'ar';
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
            vertexShader: `
                varying vec2 vUv;
                varying vec3 vPosition;
                void main() {
                    vUv = uv;
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float opacity;
                uniform vec3 color;
                uniform float pulseIntensity;
                uniform vec3 interactionPoint;
                uniform float interactionStrength;
                uniform bool isEdgeOnly;
                varying vec2 vUv;
                varying vec3 vPosition;

                void main() {
                    // Simplified pulse calculation
                    float pulse = sin(time) * 0.5 + 0.5;
                    
                    // Only calculate interaction if strength is significant
                    float interaction = 0.0;
                    if (interactionStrength > 0.01) {
                        float dist = length(vPosition - interactionPoint);
                        interaction = interactionStrength * (1.0 - smoothstep(0.0, 2.0, dist));
                    }
                    
                    float alpha;
                    if (isEdgeOnly) {
                        alpha = opacity * (0.8 + pulse * pulseIntensity + interaction);
                        vec3 edgeColor = color + vec3(0.1) * pulse; // Reduced edge brightness
                        gl_FragColor = vec4(edgeColor, clamp(alpha, 0.0, 1.0));
                    } else {
                        alpha = opacity * (0.5 + pulse * pulseIntensity + interaction);
                        gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
                    }
                }
            `,
            transparent: true,
            side: isAR ? 0 : 2, // THREE.FrontSide = 0, THREE.DoubleSide = 2
            blending: AdditiveBlending,
            wireframe: true,
            wireframeLinewidth: 1
        });

        // Set update frequency based on context
        this.updateFrequency = isAR ? 2 : 1; // Update every frame in desktop, every other frame in AR
        this.frameCount = 0;

        // Validate shader compilation if we have a renderer
        if (HologramShaderMaterial.renderer) {
            this.validateShader();
        }

        logger.debug('HologramShaderMaterial initialized', { updateFrequency: this.updateFrequency });
    }

    public static setRenderer(renderer: WebGLRenderer): void {
        HologramShaderMaterial.renderer = renderer;
        logger.debug('Renderer set for shader validation');
    }

    private validateShader(): void {
        if (!HologramShaderMaterial.renderer) {
            logger.debug('No renderer available for shader validation');
            return;
        }

        const gl = HologramShaderMaterial.renderer.domElement.getContext('webgl2') || HologramShaderMaterial.renderer.domElement.getContext('webgl');
        if (!gl) {
            logger.error('Could not get WebGL context');
            return;
        }

        const program = (this as any).program;
        if (!program) {
            logger.error('No shader program available');
            return;
        }

        const isValid = gl.getProgramParameter(program, gl.LINK_STATUS);
        if (!isValid) {
            const info = gl.getProgramInfoLog(program);
            logger.error('Shader program validation failed:', info);
        } else {
            logger.debug('Shader program validated successfully');
        }
    }

    update(deltaTime: number): void {
        this.frameCount++;
        if (this.frameCount % this.updateFrequency === 0) {
            this.uniforms.time.value += deltaTime;
            if (this.uniforms.interactionStrength.value > 0.01) {
                this.uniforms.interactionStrength.value *= 0.95; // Decay interaction effect
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
        logger.debug('Cloning HologramShaderMaterial');
        // Create settings object from current uniforms
        const settings = {
            visualization: {
                hologram: {
                    opacity: this.uniforms.opacity.value,
                    color: '#' + Array.from(this.uniforms.color.value.toArray())
                        .map(v => Math.round(v * 255).toString(16).padStart(2, '0'))
                        .join('')
                }
            }
        };
        logger.debug('Clone settings', settings);
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
        material.validateShader();
        logger.debug('Material cloned successfully');
        return material as this;
    }
}
