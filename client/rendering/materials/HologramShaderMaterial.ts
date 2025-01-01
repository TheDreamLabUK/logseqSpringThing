import { ShaderMaterial, Color, UniformsUtils, UniformsLib } from 'three';

export interface HologramShaderMaterialParameters {
    color?: Color;
    opacity?: number;
    glowIntensity?: number;
    wireframe?: boolean;
}

export class HologramShaderMaterial extends ShaderMaterial {
    declare isHologramShaderMaterial: true;

    constructor(parameters: HologramShaderMaterialParameters = {}) {
        const uniforms = UniformsUtils.merge([
            UniformsLib.common,
            UniformsLib.lights,
            {
                color: { value: parameters.color || new Color(0x00ff00) },
                opacity: { value: parameters.opacity !== undefined ? parameters.opacity : 1.0 },
                glowIntensity: { value: parameters.glowIntensity !== undefined ? parameters.glowIntensity : 1.0 },
                time: { value: 0.0 }
            }
        ]);

        super({
            uniforms,
            vertexShader: `
                uniform float time;
                uniform float glowIntensity;
                
                varying vec2 vUv;
                varying vec3 vNormal;
                varying vec3 vViewPosition;

                void main() {
                    vUv = uv;
                    vNormal = normalize(normalMatrix * normal);
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    vViewPosition = -mvPosition.xyz;
                    
                    // Add wave effect based on glowIntensity
                    vec3 pos = position;
                    float wave = sin(pos.y * 5.0 + time) * 0.05 * glowIntensity;
                    pos.x += wave;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform float opacity;
                uniform float time;
                uniform float glowIntensity;
                
                varying vec2 vUv;
                varying vec3 vNormal;
                varying vec3 vViewPosition;

                void main() {
                    // Fresnel effect
                    vec3 normal = normalize(vNormal);
                    vec3 viewDir = normalize(vViewPosition);
                    float fresnel = dot(normal, viewDir);
                    fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
                    
                    // Scan line effect with glowIntensity
                    float scanLine = sin(vUv.y * 50.0 + time * 2.0) * 0.5 + 0.5;
                    scanLine *= glowIntensity;
                    
                    // Edge highlight with glowIntensity
                    float edge = 1.0 - dot(normal, viewDir);
                    edge = pow(edge, 3.0) * glowIntensity;
                    
                    // Pulse effect
                    float pulse = sin(time) * 0.5 + 0.5;
                    
                    vec3 finalColor = color + vec3(0.1) * scanLine + vec3(0.2) * edge;
                    finalColor += color * pulse * glowIntensity * 0.2;
                    
                    float finalOpacity = opacity * (0.5 + 0.5 * fresnel + 0.2 * scanLine);
                    
                    gl_FragColor = vec4(finalColor, finalOpacity);
                }
            `,
            transparent: true,
            wireframe: parameters.wireframe || false,
            lights: true
        });

        this.isHologramShaderMaterial = true;
    }

    public update(time: number): void {
        this.uniforms.time.value = time;
    }

    public setGlowIntensity(intensity: number): void {
        this.uniforms.glowIntensity.value = intensity;
    }

    public setColor(color: Color): void {
        this.uniforms.color.value = color;
    }

    public setOpacity(opacity: number): void {
        this.uniforms.opacity.value = opacity;
    }

    public clone(): this {
        return new HologramShaderMaterial({
            color: this.uniforms.color.value.clone(),
            opacity: this.uniforms.opacity.value,
            glowIntensity: this.uniforms.glowIntensity.value,
            wireframe: this.wireframe
        }) as this;
    }
}
