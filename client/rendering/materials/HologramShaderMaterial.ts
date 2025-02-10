import * as THREE from 'three';

export interface HologramUniforms {
    [key: string]: { value: any };
    time: { value: number };
    opacity: { value: number };
    color: { value: THREE.Color };
    pulseIntensity: { value: number };
    interactionPoint: { value: THREE.Vector3 };
    interactionStrength: { value: number };
    isEdgeOnly: { value: boolean };
}

export class HologramShaderMaterial extends THREE.ShaderMaterial {
    declare uniforms: HologramUniforms;

    constructor(settings?: any, context: 'ar' | 'desktop' = 'desktop') {
        const isAR = context === 'ar';
        super({
            uniforms: {
                time: { value: 0 },
                opacity: { value: settings?.visualization?.hologram?.opacity ?? 1.0 },
                color: { value: new THREE.Color(settings?.visualization?.hologram?.color ?? 0x00ff00) },
                pulseIntensity: { value: isAR ? 0.1 : 0.2 }, // Reduced pulse intensity for AR
                interactionPoint: { value: new THREE.Vector3() },
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
            blending: THREE.AdditiveBlending,
            wireframe: true,
            wireframeLinewidth: 1
        });

        // Set update frequency based on context
        this.updateFrequency = isAR ? 2 : 1; // Update every other frame in AR
        this.frameCount = 0;
    }

    private updateFrequency: number;
    private frameCount: number;

    update(deltaTime: number): void {
        this.frameCount++;
        if (this.frameCount % this.updateFrequency === 0) {
            this.uniforms.time.value += deltaTime;
            if (this.uniforms.interactionStrength.value > 0.01) {
                this.uniforms.interactionStrength.value *= 0.95; // Decay interaction effect
            }
        }
    }

    handleInteraction(position: THREE.Vector3): void {
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
        const material = new HologramShaderMaterial();
        material.uniforms = {
            time: { value: this.uniforms.time.value },
            opacity: { value: this.uniforms.opacity.value },
            color: { value: this.uniforms.color.value.clone() },
            pulseIntensity: { value: this.uniforms.pulseIntensity.value },
            interactionPoint: { value: this.uniforms.interactionPoint.value.clone() },
            interactionStrength: { value: this.uniforms.interactionStrength.value },
            isEdgeOnly: { value: this.uniforms.isEdgeOnly.value }
        };
        return material as this;
    }
}
