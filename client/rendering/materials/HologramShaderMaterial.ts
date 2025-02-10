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

    constructor(settings?: any) {
        super({
            uniforms: {
                time: { value: 0 },
                opacity: { value: settings?.visualization?.hologram?.opacity ?? 1.0 },
                color: { value: new THREE.Color(settings?.visualization?.hologram?.color ?? 0x00ff00) },
                pulseIntensity: { value: 0.2 },
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
                    float pulse = sin(time * 2.0) * 0.5 + 0.5;
                    float dist = length(vPosition - interactionPoint);
                    float interaction = interactionStrength * (1.0 - smoothstep(0.0, 2.0, dist));
                    
                    float alpha;
                    if (isEdgeOnly) {
                        // Edge-only mode: stronger glow effect
                        alpha = opacity * (0.8 + pulse * pulseIntensity * 1.5 + interaction);
                        // Add edge enhancement
                        vec3 edgeColor = color + vec3(0.2) * pulse; // Slightly brighter edges
                        gl_FragColor = vec4(edgeColor, clamp(alpha, 0.0, 1.0));
                    } else {
                        alpha = opacity * (0.5 + pulse * pulseIntensity + interaction);
                        gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
                    }
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
        });
    }

    update(deltaTime: number): void {
        this.uniforms.time.value += deltaTime;
        this.uniforms.interactionStrength.value *= 0.95; // Decay interaction effect
    }

    handleInteraction(position: THREE.Vector3): void {
        this.uniforms.interactionPoint.value.copy(position);
        this.uniforms.interactionStrength.value = 1.0;
    }

    setEdgeOnly(enabled: boolean): void {
        this.uniforms.isEdgeOnly.value = enabled;
        this.uniforms.pulseIntensity.value = enabled ? 0.3 : 0.2; // Stronger pulse for edges
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
