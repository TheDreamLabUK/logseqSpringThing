import {
    Scene,
    Camera,
    Group,
    Texture,
    BufferGeometry,
    NearestFilter,
    ClampToEdgeWrapping,
    InstancedBufferAttribute,
    PlaneGeometry,
    Mesh,
    Vector3,
    Color,
    MeshBasicMaterial,
    BufferAttribute
} from 'three';
import { debugState } from '../core/debugState';
import { createLogger } from '../core/logger';
import { LabelSettings } from '../types/settings';
import { platformManager } from '../platform/platformManager';
import { SDFFontAtlasGenerator } from './SDFFontAtlasGenerator';
import '../types/three-ext.d';

const logger = createLogger('UnifiedTextRenderer');

// Note: Using fallback basic material approach instead of custom shaders
// to avoid WebGL shader compilation issues

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
    private material: MeshBasicMaterial;
    private geometry: BufferGeometry;
    private mesh: Mesh;
    private fontAtlas: Texture | null;
    private labels: Map<string, LabelInstance>;
    private settings: LabelSettings;
    private maxInstances: number;
    private currentInstanceCount: number;
    private logger = createLogger('UnifiedTextRenderer');
    private fontAtlasGenerator: SDFFontAtlasGenerator;
    // Reduced LABEL_SCALE by 10x
    private readonly LABEL_SCALE = 0.05; // Was 0.5 previously
    
    constructor(camera: Camera, scene: Scene, settings: LabelSettings) {
        this.scene = scene;
        this.camera = camera;
        this.settings = settings;
        
        // Only log detailed settings when data debugging is enabled
        if (debugState.isDataDebugEnabled()) {
            logger.info('UnifiedTextRenderer settings:', {
                enableLabels: this.settings.enableLabels,
                desktopFontSize: this.settings.desktopFontSize,
                textColor: this.settings.textColor,
                billboardMode: this.settings.billboardMode
            });
        }

        this.labels = new Map();
        this.maxInstances = 2000;
        this.currentInstanceCount = 0;
        this.fontAtlas = null;
        
        this.group = new Group();
        this.scene.add(this.group);
        
        this.fontAtlasGenerator = new SDFFontAtlasGenerator(2048, 8, 16);

        // Only log initialization details when debugging is enabled
        if (debugState.isDataDebugEnabled()) {
            this.logger.info('Initializing material with basic settings', {
                color: this.settings.textColor,
                transparent: true
            });
        }
        
        // Use basic material instead of shader material to avoid WebGL issues.
        // Disable depthTest and depthWrite so labels are always visible.
        this.material = new MeshBasicMaterial({ 
            color: new Color(this.settings.textColor),
            transparent: true,
            depthTest: false,
            depthWrite: false
        });
        
        this.geometry = this.createInstancedGeometry();
        
        // Only log geometry details when debugging is enabled
        if (debugState.isDataDebugEnabled()) {
            this.logger.info('Created instanced geometry:', {
                maxInstances: this.maxInstances,
                instancePosition: this.geometry.getAttribute('instancePosition')?.count,
                instanceColor: this.geometry.getAttribute('instanceColor')?.count,
                instanceScale: this.geometry.getAttribute('instanceScale')?.count
            });
        }
        
        this.mesh = new Mesh(this.geometry, this.material);
        // Ensure text labels render on top by assigning a very high render order
        this.mesh.renderOrder = 1000; // Increased to match NodeMetadataManager sprite renderOrder
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
            // Only log font atlas generation details when debugging is enabled
            if (debugState.isDataDebugEnabled()) {
                this.logger.info('Starting font atlas generation with params:', {
                    fontFamily: 'Arial',
                    fontSize: 32,
                    textureSize: (this.fontAtlasGenerator as any)['atlasSize'],
                    padding: (this.fontAtlasGenerator as any)['padding'],
                    spread: (this.fontAtlasGenerator as any)['spread']
                });
            }

            const { texture } = await this.fontAtlasGenerator.generateAtlas(
                'Arial',
                32 // Base font size for SDF
            );
            
            // Configure texture parameters
            texture.minFilter = NearestFilter;
            texture.magFilter = NearestFilter;
            texture.wrapS = ClampToEdgeWrapping;
            texture.wrapT = ClampToEdgeWrapping;
            
            this.fontAtlas = texture;
            
            // Only log atlas generation completion when debugging is enabled
            if (debugState.isDataDebugEnabled()) {
                this.logger.info('Font atlas generated successfully:', {
                    textureWidth: (texture as any).image?.width,
                    textureHeight: (texture as any).image?.height,
                    format: (texture as any).format,
                    mipmaps: (texture as any).mipmaps?.length || 0
                });
            }
            
            // Update all existing labels
            this.labels.forEach((label, id) => {
                this.updateLabel(id, label.text, label.position, label.color);
            });
        } catch (error) {
            this.logger.error('Failed to initialize font atlas:', {
                error,
                message: error instanceof Error ? error.message : String(error),
                stack: error instanceof Error ? error.stack : undefined
            });
        }
    }
    
    private createInstancedGeometry(): BufferGeometry {
        const baseGeometry = new PlaneGeometry(1, 1);
        const instancedGeometry = new BufferGeometry();
        
        // Copy attributes from base geometry
        const position = baseGeometry.getAttribute('position');
        const uv = baseGeometry.getAttribute('uv');
        const normal = baseGeometry.getAttribute('normal');
        
        instancedGeometry.setAttribute('position', position);
        instancedGeometry.setAttribute('uv', uv);
        if (normal) instancedGeometry.setAttribute('normal', normal);
        
        // Set up instanced attributes with proper sizes
        const instancePositions = new Float32Array(this.maxInstances * 3); // vec3
        const instanceColors = new Float32Array(this.maxInstances * 4);    // vec4
        const instanceScales = new Float32Array(this.maxInstances);        // float
        
        // Initialize instance attributes with proper itemSize
        instancedGeometry.setAttribute(
            'instancePosition',
            new InstancedBufferAttribute(instancePositions, 3, false)
        );
        instancedGeometry.setAttribute(
            'instanceColor',
            new InstancedBufferAttribute(instanceColors, 4, false)
        );
        instancedGeometry.setAttribute(
            'instanceScale',
            new InstancedBufferAttribute(instanceScales, 1, false)
        );
        
        // Copy index if present
        const index = (baseGeometry as any).index;
        if (index instanceof BufferAttribute) {
            instancedGeometry.setIndex(index);
        }
        
        // Clean up base geometry
        baseGeometry.dispose();
                
        return instancedGeometry;
    }
    
    public updateLabel(id: string, text: string, position: Vector3, color?: Color, preserveText: boolean = false): void {
        // Skip empty text (reduces debug spam)
        // BUT only if we don't have the preserveText flag set
        if ((!text || text.trim() === '') && !preserveText) {
            return;
        }
        
        const isPositionUpdateOnly = (!text || text.trim() === '') && preserveText;

        
        // Only log when debug is enabled
        if (debugState.isDataDebugEnabled()) {
            this.logger.debug('Updating label:', {
                id,
                text,
                position,
                color: color ? [(color as any).r, (color as any).g, (color as any).b] : undefined,
                hasAtlas: !!this.fontAtlas
            });
        }
        
        let label = this.labels.get(id);
        
        if (!label) {
            if (this.currentInstanceCount >= this.maxInstances) {
                this.logger.warn(`Maximum instance count (${this.maxInstances}) reached, cannot add more labels`);
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
            
            if (debugState.isDataDebugEnabled()) {
                this.logger.debug('Created new label instance:', {
                    id,
                    instanceIndex: this.currentInstanceCount,
                    position,
                    color: color ? [(color as any).r, (color as any).g, (color as any).b] : undefined
                });
            }
            
            this.labels.set(id, label);
            this.currentInstanceCount++;
        } else {
            // Only update text if we're not just updating the position
            if (!isPositionUpdateOnly) {
                label.text = text;
            }
            label.position.copy(position);
            if (color) label.color = color;
        }
        
        this.updateInstanceAttributes();
    }
    
    private updateInstanceAttributes(): void {
        const positions = (this.geometry.getAttribute('instancePosition') as InstancedBufferAttribute).array as Float32Array;
        const colors = (this.geometry.getAttribute('instanceColor') as InstancedBufferAttribute).array as Float32Array;
        const scales = (this.geometry.getAttribute('instanceScale') as InstancedBufferAttribute).array as Float32Array;

        if (debugState.isDataDebugEnabled()) {
            // Debug log instance updates
            this.logger.debug('Updating instance attributes:', {
                currentInstanceCount: this.currentInstanceCount,
                labelsCount: this.labels.size,
                positionsLength: positions.length,
                colorsLength: colors.length
            });
        }
        
        let index = 0;
        this.labels.forEach(label => {
            if (label.visible) {
                positions[index * 3] = label.position.x;
                positions[index * 3 + 1] = label.position.y;
                positions[index * 3 + 2] = label.position.z;
                
                const colorArray = label.color.toArray();
                colors.set(colorArray, index * 4);
                colors[index * 4 + 3] = 1.0;
                
                scales[index] = label.scale * this.LABEL_SCALE; // Apply LABEL_SCALE here
                index++;
            }
        });
        
        // Set instance count on the mesh
        (this.mesh as any).instanceCount = this.currentInstanceCount;
        
        if (debugState.isDataDebugEnabled()) {
            // Debug log final state
            this.logger.debug('Instance attributes updated:', {
                instanceCount: (this.mesh as any).instanceCount,
                visibleLabels: index
            });
        }
        
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
        if (!this.camera || !this.material) return;
        
        // When in camera billboard mode, update all labels regardless of position
        if (this.settings.billboardMode === 'camera') {
            this.labels.forEach((label, id) => {
                this.updateLabel(id, label.text, label.position, label.color);
            });
        } else {
            // For other billboard modes, update only visible labels
            this.labels.forEach((label, id) => {
                if (this.isLabelVisible(label)) {
                    this.updateLabel(id, label.text, label.position, label.color);
                }
            });
        }
    }

    private isLabelVisible(label: LabelInstance): boolean {
        if (!label.visible) return false;
        
        // When billboard_mode is "camera", don't cull labels based on position relative to camera
        // This ensures labels are visible regardless of which side of the origin they're on
        if (this.settings.billboardMode === 'camera') {
            return true;
        }
        
        // For other billboard modes, use distance-based culling with the camera's far plane
        const distanceToCamera = label.position.distanceTo(this.camera.position);
        const margin = 5.0;  // Units in world space
        
        // Check if label is within camera's view distance (with margin)
        return distanceToCamera <= (this.camera as any).far + margin;
    }

    public dispose(): void {
        this.geometry.dispose();
        if (this.material) {
            this.material.dispose();
        }
        if (this.fontAtlas) {
            this.fontAtlas.dispose();
        }
        if (this.group && this.group.parent) {
            this.group.parent.remove(this.group);
        }
    }
}
