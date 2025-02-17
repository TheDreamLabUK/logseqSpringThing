import {
    Scene,
    Camera,
    Group,
    Texture,
    ShaderMaterial,
    BufferGeometry,
    InstancedBufferAttribute,
    PlaneGeometry,
    Mesh,
    Vector3,
    Color
} from 'three';
import { createLogger } from '../core/logger';
import { LabelSettings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
import { SDFFontAtlasGenerator } from './SDFFontAtlasGenerator';
import '../types/three-ext';

const logger = createLogger('UnifiedTextRenderer');

// Vertex shader for SDF text rendering with instancing
const vertexShader = `
    attribute vec4 textureCoord;
    attribute vec3 instancePosition;
    attribute vec4 instanceColor;
    attribute float instanceScale;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    
    void main() {
        vUv = uv;
        vColor = instanceColor;
        vScale = instanceScale;
        
        vec4 mvPosition = modelViewMatrix * vec4(instancePosition, 1.0);
        vec3 scale = vec3(instanceScale);
        vec3 vertexPosition = position * scale;
        
        #ifdef BILLBOARD_VERTICAL
            float angle = atan(mvPosition.x, mvPosition.z);
            mat4 billboardMatrix = mat4(
                cos(angle), 0.0, sin(angle), 0.0,
                0.0, 1.0, 0.0, 0.0,
                -sin(angle), 0.0, cos(angle), 0.0,
                0.0, 0.0, 0.0, 1.0
            );
            mvPosition = modelViewMatrix * billboardMatrix * vec4(vertexPosition + instancePosition, 1.0);
        #else
            mvPosition = modelViewMatrix * vec4(vertexPosition + instancePosition, 1.0);
            mvPosition.xyz += position * scale;
        #endif
        
        gl_Position = projectionMatrix * mvPosition;
    }
`;

// Fragment shader for SDF text rendering
const fragmentShader = `
    uniform sampler2D fontAtlas;
    uniform float sdfThreshold;
    uniform float sdfSpread;
    uniform vec3 outlineColor;
    uniform float outlineWidth;
    
    varying vec2 vUv;
    varying vec4 vColor;
    varying float vScale;
    
    float median(float r, float g, float b) {
        return max(min(r, g), min(max(r, g), b));
    }
    
    void main() {
        vec3 sample = texture2D(fontAtlas, vUv).rgb;
        float sigDist = median(sample.r, sample.g, sample.b);
        
        float alpha = smoothstep(sdfThreshold - sdfSpread, sdfThreshold + sdfSpread, sigDist);
        float outline = smoothstep(sdfThreshold - outlineWidth - sdfSpread, 
                                 sdfThreshold - outlineWidth + sdfSpread, sigDist);
        
        vec4 color = mix(vec4(outlineColor, outline), vColor, alpha);
        gl_FragColor = color;
    }
`;

interface LabelInstance {
    id: string;
    text: string;
    position: Vector3;
    scale: number;
    color: Color;
    visible: boolean;
}

export class UnifiedTextRenderer {
    private scene: Scene;
    private camera: Camera;
    private group: Group;
    private material: ShaderMaterial;
    private geometry: BufferGeometry;
    private mesh: Mesh;
    private fontAtlas: Texture | null;
    private labels: Map<string, LabelInstance>;
    private settings: LabelSettings;
    private maxInstances: number;
    private currentInstanceCount: number;
    private fontAtlasGenerator: SDFFontAtlasGenerator;
    
    constructor(camera: Camera, scene: Scene, settings: LabelSettings) {
        this.scene = scene;
        this.camera = camera;
        this.settings = settings;
        this.labels = new Map();
        this.maxInstances = 1000;
        this.currentInstanceCount = 0;
        this.fontAtlas = null;
        
        this.group = new Group();
        this.scene.add(this.group);
        
        this.fontAtlasGenerator = new SDFFontAtlasGenerator(1024, 4, 8);
        
        // Initialize the material with a temporary texture
        this.material = new ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                fontAtlas: { value: null },
                sdfThreshold: { value: 0.5 },
                sdfSpread: { value: 0.1 },
                outlineColor: { value: new Color(settings.textOutlineColor) },
                outlineWidth: { value: settings.textOutlineWidth }
            },
            transparent: true,
            depthTest: true,
            defines: {
                BILLBOARD_VERTICAL: settings.billboardMode === 'vertical'
            }
        });
        
        this.geometry = this.createInstancedGeometry();
        this.mesh = new Mesh(this.geometry, this.material);
        this.group.add(this.mesh);
        
        this.setXRMode(platformManager.isXRMode);
        platformManager.on('xrmodechange', (enabled: boolean) => {
            this.setXRMode(enabled);
        });
        
        // Initialize font atlas
        this.initializeFontAtlas();
    }
    
    private async initializeFontAtlas(): Promise<void> {
        try {
            const { texture } = await this.fontAtlasGenerator.generateAtlas(
                'Arial',
                32 // Base font size for SDF
            );
            
            this.fontAtlas = texture;
            this.material.uniforms.fontAtlas.value = texture;
            
            // Update all existing labels
            this.labels.forEach((label, id) => {
                this.updateLabel(id, label.text, label.position, label.color);
            });
        } catch (error) {
            logger.error('Failed to initialize font atlas:', error);
        }
    }
    
    private createInstancedGeometry(): BufferGeometry {
        const baseGeometry = new PlaneGeometry(1, 1);
        const instancedGeometry = new BufferGeometry();
        
        const position = baseGeometry.getAttribute('position');
        const uv = baseGeometry.getAttribute('uv');
        instancedGeometry.setAttribute('position', position);
        instancedGeometry.setAttribute('uv', uv);
        
        const instancePositions = new Float32Array(this.maxInstances * 3);
        const instanceColors = new Float32Array(this.maxInstances * 4);
        const instanceScales = new Float32Array(this.maxInstances);
        
        instancedGeometry.setAttribute(
            'instancePosition',
            new InstancedBufferAttribute(instancePositions, 3)
        );
        instancedGeometry.setAttribute(
            'instanceColor',
            new InstancedBufferAttribute(instanceColors, 4)
        );
        instancedGeometry.setAttribute(
            'instanceScale',
            new InstancedBufferAttribute(instanceScales, 1)
        );
        
        return instancedGeometry;
    }
    
    public updateLabel(id: string, text: string, position: Vector3, color?: Color): void {
        let label = this.labels.get(id);
        
        if (!label) {
            if (this.currentInstanceCount >= this.maxInstances) {
                logger.warn('Maximum instance count reached');
                return;
            }
            
            label = {
                id,
                text,
                position: position.clone(),
                scale: 1.0,
                color: color || new Color(this.settings.textColor),
                visible: true
            };
            
            this.labels.set(id, label);
            this.currentInstanceCount++;
        } else {
            label.text = text;
            label.position.copy(position);
            if (color) label.color = color;
        }
        
        this.updateInstanceAttributes();
    }
    
    private updateInstanceAttributes(): void {
        const positions = (this.geometry.getAttribute('instancePosition') as InstancedBufferAttribute).array as Float32Array;
        const colors = (this.geometry.getAttribute('instanceColor') as InstancedBufferAttribute).array as Float32Array;
        const scales = (this.geometry.getAttribute('instanceScale') as InstancedBufferAttribute).array as Float32Array;
        
        let index = 0;
        this.labels.forEach(label => {
            if (label.visible) {
                positions[index * 3] = label.position.x;
                positions[index * 3 + 1] = label.position.y;
                positions[index * 3 + 2] = label.position.z;
                
                const colorArray = label.color.toArray();
                colors.set(colorArray, index * 4);
                colors[index * 4 + 3] = 1.0;
                
                scales[index] = label.scale;
                index++;
            }
        });
        
        (this.geometry.getAttribute('instancePosition') as InstancedBufferAttribute).needsUpdate = true;
        (this.geometry.getAttribute('instanceColor') as InstancedBufferAttribute).needsUpdate = true;
        (this.geometry.getAttribute('instanceScale') as InstancedBufferAttribute).needsUpdate = true;
    }
    
    public removeLabel(id: string): void {
        if (this.labels.delete(id)) {
            this.currentInstanceCount--;
            this.updateInstanceAttributes();
        }
    }
    
    public setXRMode(enabled: boolean): void {
        if (enabled) {
            this.group.layers.disable(0);
            this.group.layers.enable(1);
        } else {
            this.group.layers.enable(0);
            this.group.layers.enable(1);
        }
    }
    
    public update(): void {
        this.camera.updateMatrixWorld();
    }
    
    public dispose(): void {
        this.geometry.dispose();
        this.material.dispose();
        if (this.fontAtlas) {
            this.fontAtlas.dispose();
        }
        if (this.group.parent) {
            this.group.parent.remove(this.group);
        }
    }
}