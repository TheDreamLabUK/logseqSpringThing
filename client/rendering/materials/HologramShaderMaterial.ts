import * as THREE from 'three';

export interface HologramUniforms {
    [key: string]: { value: any };
    time: { value: number };
    opacity: { value: number };
    color: { value: THREE.Color };
    pulseIntensity: { value: number };
    interactionPoint: { value: THREE.Vector3 };
    interactionStrength: { value: number };
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
                interactionStrength: { value: 0.0 }
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
                varying vec2 vUv;
                varying vec3 vPosition;

                void main() {
                    float pulse = sin(time * 2.0) * 0.5 + 0.5;
                    float dist = length(vPosition - interactionPoint);
                    float interaction = interactionStrength * (1.0 - smoothstep(0.0, 2.0, dist));
                    float alpha = opacity * (0.5 + pulse * pulseIntensity + interaction);
                    gl_FragColor = vec4(color, alpha);
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

    clone(): this {
        const material = new HologramShaderMaterial();
        material.uniforms = {
            time: { value: this.uniforms.time.value },
            opacity: { value: this.uniforms.opacity.value },
            color: { value: this.uniforms.color.value.clone() },
            pulseIntensity: { value: this.uniforms.pulseIntensity.value },
            interactionPoint: { value: this.uniforms.interactionPoint.value.clone() },
            interactionStrength: { value: this.uniforms.interactionStrength.value }
        };
        return material as this;
    }
}
