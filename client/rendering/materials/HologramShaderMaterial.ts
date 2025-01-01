import * as THREE from 'three';
import { HologramUniforms, HologramShaderMaterial as IHologramShaderMaterial } from '../../core/types';

export interface HologramShaderMaterialParameters {
    color?: THREE.Color;
    opacity?: number;
    glowIntensity?: number;
}

export class HologramShaderMaterial extends THREE.ShaderMaterial implements IHologramShaderMaterial {
    declare uniforms: HologramUniforms;

    constructor(parameters: HologramShaderMaterialParameters = {}) {
        super();
        
        this.uniforms = {
            time: { value: 0 },
            opacity: { value: parameters.opacity ?? 1.0 },
            color: { value: parameters.color ?? new THREE.Color() },
            glowIntensity: { value: parameters.glowIntensity ?? 0.5 }
        };

        this.vertexShader = `
            varying vec2 vUv;
            varying vec3 vPosition;
            void main() {
                vUv = uv;
                vPosition = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        this.fragmentShader = `
            uniform float time;
            uniform float opacity;
            uniform vec3 color;
            uniform float glowIntensity;
            varying vec2 vUv;
            varying vec3 vPosition;

            void main() {
                float glow = sin(time * 2.0) * 0.5 + 0.5;
                vec3 finalColor = color * (1.0 + glow * glowIntensity);
                gl_FragColor = vec4(finalColor, opacity);
            }
        `;

        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.lights = true;
    }

    update(deltaTime: number): void {
        this.uniforms.time.value += deltaTime;
    }

    handleInteraction(intensity: number): void {
        this.uniforms.glowIntensity.value = intensity;
    }
}
