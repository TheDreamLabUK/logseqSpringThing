import * as THREE from 'three';

export interface HologramUniforms {
    time: { value: number };
    opacity: { value: number };
    color: { value: THREE.Color };
}

export class HologramShaderMaterial extends THREE.ShaderMaterial {
    declare uniforms: HologramUniforms;

    constructor(settings?: {
        visualization?: {
            hologram?: {
                opacity?: number;
                color?: number;
            };
        };
    }) {
        const uniforms: HologramUniforms = {
            time: { value: 0 },
            opacity: { value: settings?.visualization?.hologram?.opacity ?? 0.5 },
            color: { value: new THREE.Color(settings?.visualization?.hologram?.color ?? 0x00ff00) }
        };

        super({
            uniforms,
            vertexShader: `
                uniform float time;
                varying vec2 vUv;
                varying float vIntensity;

                void main() {
                    vUv = uv;
                    vec3 pos = position;
                    pos.y += sin(pos.x * 10.0 + time) * 0.1;
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    vIntensity = 1.0 - abs(normalize(normalMatrix * normal).z);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform float opacity;
                varying vec2 vUv;
                varying float vIntensity;

                void main() {
                    vec3 glow = color * vIntensity;
                    gl_FragColor = vec4(glow, opacity);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide
        });
    }

    update(time: number): void {
        this.uniforms.time.value = time;
    }
}
