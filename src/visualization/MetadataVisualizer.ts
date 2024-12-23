import * as THREE from 'three';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry';
import { Font } from 'three/examples/jsm/loaders/FontLoader';
import { NodeMetadata } from '../types/metadata';

export class MetadataVisualizer {
    private static readonly SHAPES = {
        SPHERE: new THREE.SphereGeometry(1, 32, 32),
        DODECAHEDRON: new THREE.DodecahedronGeometry(1),
        ICOSAHEDRON: new THREE.IcosahedronGeometry(1),
        OCTAHEDRON: new THREE.OctahedronGeometry(1)
    };

    private readonly font: Font;
    private readonly textMaterial: THREE.ShaderMaterial;
    private readonly labelGroup: THREE.Group;

    constructor(
        private readonly scene: THREE.Scene,
        private readonly camera: THREE.Camera,
        private readonly settings: any
    ) {
        this.labelGroup = new THREE.Group();
        scene.add(this.labelGroup);

        // Initialize SDF text shader
        this.textMaterial = new THREE.ShaderMaterial({
            uniforms: {
                uColor: { value: new THREE.Color(settings.labels.text_color) },
                uOutlineColor: { value: new THREE.Color(settings.labels.text_outline_color) },
                uOutlineWidth: { value: settings.labels.text_outline_width },
                uOpacity: { value: 1.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform vec3 uColor;
                uniform vec3 uOutlineColor;
                uniform float uOutlineWidth;
                uniform float uOpacity;
                varying vec2 vUv;
                
                void main() {
                    float distance = texture2D(uMap, vUv).a;
                    float alpha = smoothstep(0.5 - uOutlineWidth, 0.5 + uOutlineWidth, distance);
                    vec3 color = mix(uOutlineColor, uColor, smoothstep(0.5 - uOutlineWidth, 0.5, distance));
                    gl_FragColor = vec4(color, alpha * uOpacity);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
    }

    public createNodeVisual(metadata: NodeMetadata): THREE.Mesh {
        const geometry = this.getGeometryFromAge(metadata.commitAge);
        const material = this.createMaterialFromHyperlinks(metadata.hyperlinkCount);
        const mesh = new THREE.Mesh(geometry, material);
        
        // Set size from metadata
        const scale = this.calculateNodeScale(metadata.importance);
        mesh.scale.setScalar(scale);

        // Add label
        this.createLabel(mesh, metadata.name);

        return mesh;
    }

    private getGeometryFromAge(ageInDays: number): THREE.BufferGeometry {
        const ranges = this.settings.nodes.shape_age_ranges;
        
        if (ageInDays <= ranges[0]) return MetadataVisualizer.SHAPES.SPHERE;
        if (ageInDays <= ranges[1]) return MetadataVisualizer.SHAPES.DODECAHEDRON;
        if (ageInDays <= ranges[2]) return MetadataVisualizer.SHAPES.ICOSAHEDRON;
        return MetadataVisualizer.SHAPES.OCTAHEDRON;
    }

    private createMaterialFromHyperlinks(linkCount: number): THREE.Material {
        const minColor = new THREE.Color(this.settings.nodes.hyperlink_color_min);
        const maxColor = new THREE.Color(this.settings.nodes.hyperlink_color_max);
        
        // Normalize link count (assuming max of 100 links)
        const t = Math.min(linkCount / 100, 1);
        const color = new THREE.Color().lerpColors(minColor, maxColor, t);

        return new THREE.MeshStandardMaterial({
            color: color,
            metalness: this.settings.nodes.metalness,
            roughness: this.settings.nodes.roughness,
            transparent: true,
            opacity: this.settings.nodes.opacity
        });
    }

    private calculateNodeScale(importance: number): number {
        const [min, max] = this.settings.nodes.size_range;
        return min + (max - min) * Math.min(importance, 1);
    }

    private createLabel(nodeMesh: THREE.Mesh, text: string): void {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;
        
        // Configure for SDF rendering
        canvas.width = this.settings.labels.text_resolution;
        canvas.height = this.settings.labels.text_resolution;
        
        ctx.font = `${this.settings.labels.desktop_font_size}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Draw text with padding for SDF
        ctx.fillText(text, 
            canvas.width / 2, 
            canvas.height / 2, 
            canvas.width - this.settings.labels.text_padding * 2
        );

        // Create texture and geometry
        const texture = new THREE.CanvasTexture(canvas);
        const geometry = new THREE.PlaneGeometry(1, 1);
        const material = this.textMaterial.clone();
        material.uniforms.uMap = { value: texture };

        const label = new THREE.Mesh(geometry, material);
        
        // Position above node
        label.position.copy(nodeMesh.position);
        label.position.y += nodeMesh.scale.y + 0.5;
        
        // Billboard behavior
        if (this.settings.labels.billboard_mode === 'camera') {
            label.onBeforeRender = () => {
                label.quaternion.copy(this.camera.quaternion);
            };
        } else {
            // Vertical billboard - only rotate around Y
            label.onBeforeRender = () => {
                const cameraPos = this.camera.position.clone();
                cameraPos.y = label.position.y;
                label.lookAt(cameraPos);
            };
        }

        this.labelGroup.add(label);
    }

    public dispose(): void {
        // Clean up geometries
        Object.values(MetadataVisualizer.SHAPES).forEach(geometry => {
            geometry.dispose();
        });

        // Clean up materials and textures
        this.labelGroup.traverse((object) => {
            if (object instanceof THREE.Mesh) {
                object.geometry.dispose();
                if (object.material instanceof THREE.Material) {
                    object.material.dispose();
                }
                if (object.material instanceof THREE.ShaderMaterial) {
                    Object.values(object.material.uniforms).forEach(uniform => {
                        if (uniform.value instanceof THREE.Texture) {
                            uniform.value.dispose();
                        }
                    });
                }
            }
        });

        // Remove from scene
        this.scene.remove(this.labelGroup);
    }
}
