import { ShaderMaterial, Color, DoubleSide } from 'three';

export class HologramMaterial extends ShaderMaterial {
    constructor() {
        super({
            vertexShader: `
                void main() {
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                void main() {
                    gl_FragColor = vec4(0.0, 1.0, 0.0, 0.5); // Hardcoded green color
                }
            `,
            transparent: true,
            side: DoubleSide
        });
    }

    public update(time: number): void {
        // No updates needed
    }
} 