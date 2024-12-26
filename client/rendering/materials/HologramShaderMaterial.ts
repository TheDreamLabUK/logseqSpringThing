import { Color, DoubleSide, Material, MaterialParameters } from 'three';

export interface HologramUniforms {
    color: { value: Color };
    opacity: { value: number };
    time: { value: number };
    pulseSpeed: { value: number };
    pulseIntensity: { value: number };
}

export class HologramShaderMaterial extends Material {
    uniforms: HologramUniforms;
    defines: { [key: string]: string | number | boolean };
    vertexShader: string;
    fragmentShader: string;
    transparent: boolean;
    side: typeof DoubleSide;
    depthWrite: boolean;
    needsUpdate: boolean;

    constructor(params: MaterialParameters & { uniforms: HologramUniforms }) {
        super();
        this.uniforms = params.uniforms;
        this.defines = {};
        this.vertexShader = `
            varying vec3 vPosition;
            varying vec2 vUv;
            void main() {
                vPosition = position;
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        this.fragmentShader = `
            uniform vec3 color;
            uniform float opacity;
            uniform float time;
            uniform float pulseSpeed;
            uniform float pulseIntensity;
            varying vec3 vPosition;
            varying vec2 vUv;
            
            void main() {
                float pulse = sin(time * pulseSpeed) * pulseIntensity + 1.0;
                float edge = 1.0 - smoothstep(0.4, 0.5, abs(vUv.y - 0.5));
                vec3 finalColor = color * pulse;
                float finalOpacity = opacity * edge;
                
                #ifdef USE_AR
                    float depth = gl_FragCoord.z / gl_FragCoord.w;
                    finalOpacity *= smoothstep(10.0, 0.0, depth);
                #endif
                
                gl_FragColor = vec4(finalColor, finalOpacity);
            }
        `;
        this.transparent = true;
        this.side = DoubleSide;
        this.depthWrite = false;
        this.needsUpdate = true;
    }

    clone(): this {
        const material = new HologramShaderMaterial({
            uniforms: {
                color: { value: new Color(this.uniforms.color.value) },
                opacity: { value: this.uniforms.opacity.value },
                time: { value: this.uniforms.time.value },
                pulseSpeed: { value: this.uniforms.pulseSpeed.value },
                pulseIntensity: { value: this.uniforms.pulseIntensity.value }
            }
        });
        return material as this;
    }

    dispose(): void {
        super.dispose();
    }
}
