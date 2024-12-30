import { Vector2, Vector3, Color, Texture, Material, DoubleSide, Side } from 'three';
import { Settings } from '../../types/settings';

interface HologramUniforms {
    time: { value: number };
    color: { value: Vector3 };
    diffuseTexture: { value: Texture | null };
    resolution: { value: Vector2 };
    interactionPoint: { value: Vector3 };
    interactionStrength: { value: number };
    opacity: { value: number };
    pulseIntensity: { value: number };
}

export class HologramShaderMaterial extends Material {
    uniforms: HologramUniforms;
    vertexShader: string;
    fragmentShader: string;
    color: Color = new Color();
    type = 'HologramShaderMaterial';
    defines: { [key: string]: string | number | boolean } = {};

    // Declare required Material properties
    declare transparent: boolean;
    declare opacity: number;
    declare depthTest: boolean;
    declare depthWrite: boolean;
    declare blending: number;
    declare side: Side;
    declare needsUpdate: boolean;
    declare vertexColors: boolean;
    declare visible: boolean;
    declare toneMapped: boolean;
    declare fog: boolean;
    declare lights: boolean;

    constructor(settings: Settings) {
        super();

        this.uniforms = {
            time: { value: 0 },
            color: { value: new Vector3() },
            diffuseTexture: { value: null },
            resolution: { value: new Vector2(window.innerWidth, window.innerHeight) },
            interactionPoint: { value: new Vector3() },
            interactionStrength: { value: 0.0 },
            opacity: { value: settings.visualization.hologram.ringOpacity },
            pulseIntensity: { value: 0.2 }
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
            #ifdef GL_ES
            precision highp float;
            #endif

            uniform float time;
            uniform vec3 color;
            uniform sampler2D diffuseTexture;
            uniform vec2 resolution;
            uniform vec3 interactionPoint;
            uniform float interactionStrength;
            uniform float opacity;
            uniform float pulseIntensity;
            varying vec2 vUv;
            varying vec3 vPosition;

            void main() {
                vec2 uv = vUv;
                
                // Base distortion
                float distortion = sin(time * 2.0 + uv.x * 10.0) * 0.02;
                
                // Interaction distortion
                float dist = length(vPosition - interactionPoint);
                float interactionDistortion = interactionStrength * (1.0 - smoothstep(0.0, 2.0, dist));
                distortion += interactionDistortion;
                
                uv.y += distortion;
                vec4 texColor = vec4(1.0); // Default white if no texture
                if (vUv.x >= 0.0) {
                    texColor = texture2D(diffuseTexture, uv);
                }
                
                // Hologram effect
                float scanline = sin(uv.y * 100.0 + time * 5.0) * 0.1 + 0.9;
                float flicker = sin(time * 20.0) * 0.05 + 0.95;
                
                // Pulse effect
                float pulse = sin(time * 3.0) * pulseIntensity;
                
                // Alpha calculation
                float alpha = opacity * scanline * flicker * (1.0 + pulse);
                if (interactionStrength > 0.0) {
                    alpha *= (1.0 + interactionDistortion);
                }
                
                gl_FragColor = vec4(color, alpha) * texColor;
            }
        `;

        // Set material properties
        this.transparent = true;
        this.depthWrite = true;
        this.depthTest = true;
        this.side = DoubleSide;
        this.blending = 1; // NormalBlending = 1
        this.vertexColors = false;
        this.visible = true;
        this.toneMapped = true;
        this.fog = false;
        this.lights = false;

        // Set initial color from settings
        const hexColor = settings.visualization.hologram.ringColor;
        const hex = hexColor.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16) / 255;
        const g = parseInt(hex.substring(2, 4), 16) / 255;
        const b = parseInt(hex.substring(4, 6), 16) / 255;
        this.uniforms.color.value.set(r, g, b);
        this.color.set(hexColor);

        this.opacity = settings.visualization.hologram.ringOpacity;
        this.uniforms.opacity.value = this.opacity;
        this.needsUpdate = true;
    }

    update(deltaTime: number): void {
        this.uniforms.time.value += deltaTime;
        this.uniforms.interactionStrength.value *= 0.95; // Gradual decay
    }

    handleInteraction(position: Vector3): void {
        this.uniforms.interactionPoint.value.copy(position);
        this.uniforms.interactionStrength.value = 1.0; // Full strength on interaction
    }

    dispose(): void {
        super.dispose();
        
        // Dispose of any textures
        if (this.uniforms.diffuseTexture.value) {
            this.uniforms.diffuseTexture.value.dispose();
        }
    }

    clone(): this {
        const material = new HologramShaderMaterial({
            visualization: {
                hologram: {
                    ringColor: '#00FFFF', // Default hologram color
                    ringOpacity: this.opacity,
                    ringCount: 3,
                    ringSizes: [1.0, 1.5, 2.0],
                    ringRotationSpeed: 0.1,
                    globalRotationSpeed: 0.05,
                    enableBuckminster: true,
                    buckminsterScale: 1.0,
                    buckminsterOpacity: 0.3,
                    enableGeodesic: true,
                    geodesicScale: 1.2,
                    geodesicOpacity: 0.4,
                    enableTriangleSphere: true,
                    triangleSphereScale: 1.1,
                    triangleSphereOpacity: 0.35
                }
            }
        } as Settings);
        material.uniforms = { ...this.uniforms };
        return material as this;
    }
}
