import { defineComponent, h, onMounted, onBeforeUnmount, watch, ref, inject, type Ref } from 'vue'
import * as THREE from 'three'
import { SCENE_KEY } from '../../composables/useVisualization'

// Define our supported event types
type SupportedEventType = 'click' | 'pointerenter' | 'pointerleave' | 'pointerdown' | 'pointermove' | 'pointerup';

// Event handler setup
const setupEventHandlers = (
  mesh: THREE.Mesh,
  emit: (event: SupportedEventType, data: THREE.Event) => void
) => {
  // Map Vue events to Three.js events
  const handlers: Record<string, (event: THREE.Event) => void> = {
    'click': (event) => {
      debugLog('Mesh', 'Event: click', { id: mesh.id });
      emit('click', event);
    },
    'pointermove': (event) => {
      debugLog('Mesh', 'Event: pointermove', { id: mesh.id });
      emit('pointermove', event);
    },
    'pointerdown': (event) => {
      debugLog('Mesh', 'Event: pointerdown', { id: mesh.id });
      emit('pointerdown', event);
    },
    'pointerup': (event) => {
      debugLog('Mesh', 'Event: pointerup', { id: mesh.id });
      emit('pointerup', event);
    },
    'pointerover': (event) => {
      debugLog('Mesh', 'Event: pointerenter', { id: mesh.id });
      emit('pointerenter', event);
    },
    'pointerout': (event) => {
      debugLog('Mesh', 'Event: pointerleave', { id: mesh.id });
      emit('pointerleave', event);
    }
  };

  // Add event listeners
  Object.entries(handlers).forEach(([type, handler]) => {
    mesh.addEventListener(type as keyof THREE.Object3DEventMap, handler);
  });

  return () => {
    // Remove event listeners on cleanup
    Object.entries(handlers).forEach(([type, handler]) => {
      mesh.removeEventListener(type as keyof THREE.Object3DEventMap, handler);
    });
  };
};

// Debug logging helper
const debugLog = (component: string, action: string, details: any) => {
  console.debug(`[Three.js ${component}] ${action}:`, details);
};

// Helper to safely get fog properties
const getFogDetails = (fog: THREE.Fog | THREE.FogExp2 | null) => {
  if (!fog) return null;
  if (fog instanceof THREE.Fog) {
    return {
      type: 'Fog',
      near: fog.near,
      far: fog.far
    };
  }
  if (fog instanceof THREE.FogExp2) {
    return {
      type: 'FogExp2',
      density: fog.density
    };
  }
  return null;
};

// Provide/inject keys
const PARENT_KEY = Symbol('three-parent');
const GEOMETRY_REF = Symbol('geometry-ref');
const MATERIAL_REF = Symbol('material-ref');

// Basic Three.js component wrapper
export const Group = defineComponent({
  name: 'Group',
  setup(_, { slots }) {
    const group = new THREE.Group();
    const parent = inject(PARENT_KEY, null) as THREE.Object3D | null;
    const scene = inject(SCENE_KEY, null) as THREE.Scene | null;
    
    debugLog('Group', 'Created', { id: group.id });
    
    onMounted(() => {
      if (parent) {
        parent.add(group);
      } else if (scene) {
        scene.add(group);
      }
      debugLog('Group', 'Mounted', { 
        id: group.id,
        parent: parent?.type || 'Scene',
        childCount: group.children.length
      });
    });

    onBeforeUnmount(() => {
      debugLog('Group', 'Disposing', { id: group.id });
      if (parent) {
        parent.remove(group);
      } else if (scene) {
        scene.remove(group);
      }
      group.clear();
    });

    return () => h('div', { provide: { [PARENT_KEY]: group } }, slots.default?.());
  }
});

export const Scene = defineComponent({
  name: 'Scene',
  props: {
    background: {
      type: [Number, String],
      default: 0x000000
    }
  },
  setup(props, { slots }) {
    // Create scene
    const scene = new THREE.Scene();
    
    // Set initial background
    scene.background = new THREE.Color(props.background);
    
    debugLog('Scene', 'Created', { 
      id: scene.id,
      background: scene.background instanceof THREE.Color ? scene.background.getHexString() : 'none'
    });

    // Watch for background changes
    watch(() => props.background, (newColor) => {
      scene.background = new THREE.Color(newColor);
      debugLog('Scene', 'Background updated', { 
        background: scene.background instanceof THREE.Color ? scene.background.getHexString() : 'none'
      });
    });

    // Clean up
    onBeforeUnmount(() => {
      debugLog('Scene', 'Disposing', { id: scene.id });
      scene.clear();
    });

    // Provide scene to child components
    return () => h('div', { provide: { [SCENE_KEY]: scene } }, slots.default?.());
  }
});

// Mesh component that wraps Three.js mesh
export const Mesh = defineComponent({
  name: 'Mesh',
  props: {
    position: {
      type: Object as () => THREE.Vector3,
      required: true
    },
    scale: {
      type: Object as () => { x: number; y: number; z: number },
      default: () => ({ x: 1, y: 1, z: 1 })
    }
  },
  emits: ['click', 'pointerenter', 'pointerleave', 'pointerdown', 'pointermove', 'pointerup'] as const,
  setup(props, { emit, slots }) {
    const mesh = new THREE.Mesh();
    const parent = inject(PARENT_KEY, null) as THREE.Object3D | null;
    const scene = inject(SCENE_KEY, null) as THREE.Scene | null;
    let cleanup: (() => void) | null = null;

    if (!parent && !scene) {
      throw new Error('Mesh must be used within a Scene or Group component');
    }

    // Wait for geometry and material from child components
    const geometryRef = ref<THREE.BufferGeometry | null>(null);
    const materialRef = ref<THREE.Material | null>(null);

    // Update mesh when geometry or material change
    watch([geometryRef, materialRef], ([geometry, material]) => {
      if (geometry && material) {
        mesh.geometry = geometry;
        mesh.material = material;
        
        // Force geometry update
        geometry.attributes.position.needsUpdate = true;
        if (geometry.index) {
          geometry.index.needsUpdate = true;
        }
        
        // Force transform update
        mesh.updateMatrix();
        mesh.updateMatrixWorld(true);
        
        // Mark for render
        if (scene?.userData) {
          scene.userData.needsRender = true;
        }

        debugLog('Mesh', 'Updated geometry and material', {
          id: mesh.id,
          geometryType: geometry.type,
          materialType: material.type,
          vertices: geometry.attributes.position?.count,
          needsUpdate: geometry.attributes.position.needsUpdate
        });
      }
    }, { immediate: true });

    onMounted(() => {
      mesh.position.copy(props.position);
      mesh.scale.set(props.scale.x, props.scale.y, props.scale.z);
      
      if (parent) {
        parent.add(mesh);
      } else if (scene) {
        scene.add(mesh);
      }

      debugLog('Mesh', 'Added to scene', {
        id: mesh.id,
        position: mesh.position.toArray(),
        scale: [mesh.scale.x, mesh.scale.y, mesh.scale.z],
        parent: parent?.type || 'Scene',
        hasGeometry: !!mesh.geometry,
        hasMaterial: !!mesh.material
      });

      // Set up event handlers
      cleanup = setupEventHandlers(mesh, emit);
    });

    // Watch for prop changes
    watch(() => props.position, (newPos) => {
      mesh.position.copy(newPos);
      debugLog('Mesh', 'Position updated', {
        id: mesh.id,
        position: mesh.position.toArray()
      });
    });

    watch(() => props.scale, (newScale) => {
      mesh.scale.set(newScale.x, newScale.y, newScale.z);
      debugLog('Mesh', 'Scale updated', {
        id: mesh.id,
        scale: [mesh.scale.x, mesh.scale.y, mesh.scale.z]
      });
    });

    onBeforeUnmount(() => {
      // Clean up event handlers
      if (cleanup) {
        cleanup();
      }

      debugLog('Mesh', 'Removing from scene', {
        id: mesh.id,
        position: mesh.position.toArray()
      });
      if (parent) {
        parent.remove(mesh);
      } else if (scene) {
        scene.remove(mesh);
      }
      mesh.geometry?.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach(m => m.dispose());
      } else if (mesh.material) {
        mesh.material.dispose();
      }
    });

    // Provide mesh to child components and handle their setup
    return () => h('div', { 
      provide: { 
        [PARENT_KEY]: mesh,
        [GEOMETRY_REF]: geometryRef,
        [MATERIAL_REF]: materialRef
      } 
    }, slots.default?.());
  }
});

// Update SphereGeometry to connect with parent mesh
export const SphereGeometry = defineComponent({
  name: 'SphereGeometry',
  props: {
    args: {
      type: Array as unknown as () => [number, number, number],
      default: () => [1, 32, 32]
    }
  },
  setup(props) {
    const geometry = new THREE.SphereGeometry(...props.args);
    const geometryRef = inject<Ref<THREE.BufferGeometry | null>>(GEOMETRY_REF);
    
    if (geometryRef) {
      geometryRef.value = geometry;
    }
    
    debugLog('SphereGeometry', 'Created', {
      radius: props.args[0],
      segments: [props.args[1], props.args[2]],
      vertices: geometry.attributes.position.count
    });

    onBeforeUnmount(() => {
      if (geometryRef) {
        geometryRef.value = null;
      }
      debugLog('SphereGeometry', 'Disposing', {
        vertices: geometry.attributes.position.count
      });
      geometry.dispose();
    });

    return { geometry };
  }
});

// Update MeshStandardMaterial to connect with parent mesh
export const MeshStandardMaterial = defineComponent({
  name: 'MeshStandardMaterial',
  props: {
    color: {
      type: String,
      default: '#ffffff'
    },
    metalness: {
      type: Number,
      default: 0.1
    },
    roughness: {
      type: Number,
      default: 0.5
    },
    opacity: {
      type: Number,
      default: 1.0
    },
    transparent: {
      type: Boolean,
      default: false
    },
    emissive: {
      type: String,
      default: '#000000'
    },
    emissiveIntensity: {
      type: Number,
      default: 1.0
    }
  },
  setup(props) {
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(props.color),
      metalness: props.metalness,
      roughness: props.roughness,
      opacity: props.opacity,
      transparent: props.transparent,
      emissive: new THREE.Color(props.emissive),
      emissiveIntensity: props.emissiveIntensity
    });

    const materialRef = inject<Ref<THREE.Material | null>>(MATERIAL_REF);
    
    if (materialRef) {
      materialRef.value = material;
    }

    debugLog('MeshStandardMaterial', 'Created', {
      color: props.color,
      metalness: props.metalness,
      roughness: props.roughness,
      opacity: props.opacity,
      emissive: props.emissive,
      emissiveIntensity: props.emissiveIntensity
    });

    watch(() => props.color, (newColor) => {
      material.color.set(newColor);
      debugLog('MeshStandardMaterial', 'Color updated', { color: newColor });
    });

    watch(() => props.emissive, (newColor) => {
      material.emissive.set(newColor);
      debugLog('MeshStandardMaterial', 'Emissive updated', { emissive: newColor });
    });

    onBeforeUnmount(() => {
      if (materialRef) {
        materialRef.value = null;
      }
      debugLog('MeshStandardMaterial', 'Disposing', {
        color: material.color.getHexString(),
        emissive: material.emissive.getHexString()
      });
      material.dispose();
    });

    return { material };
  }
});

// Line component
export const Line = defineComponent({
  name: 'Line',
  props: {
    points: {
      type: Array as unknown as () => [THREE.Vector3, THREE.Vector3],
      required: true,
      validator: (value: unknown) => {
        if (!Array.isArray(value) || value.length !== 2) return false;
        return value.every(v => v instanceof THREE.Vector3);
      }
    },
    color: {
      type: String,
      default: '#ffffff'
    },
    linewidth: {
      type: Number,
      default: 1
    },
    opacity: {
      type: Number,
      default: 1
    },
    transparent: {
      type: Boolean,
      default: false
    }
  },
  setup(props) {
    const scene = inject(SCENE_KEY) as THREE.Scene | null;
    const parent = inject(PARENT_KEY, null) as THREE.Object3D | null;

    if (!scene && !parent) {
      throw new Error('Line must be used within a Scene or Group component');
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(props.points);
    const material = new THREE.LineBasicMaterial({
      color: props.color,
      linewidth: props.linewidth,
      opacity: props.opacity,
      transparent: props.transparent
    });
    const line = new THREE.Line(geometry, material);

    onMounted(() => {
      if (parent) {
        parent.add(line);
      } else if (scene) {
        scene.add(line);
      }

      debugLog('Line', 'Added to scene', {
        points: props.points.map(p => p.toArray()),
        color: props.color,
        linewidth: props.linewidth,
        opacity: props.opacity,
        parent: parent?.type || 'Scene'
      });
    });

    watch(() => props.points, (newPoints) => {
      geometry.setFromPoints(newPoints);
      geometry.attributes.position.needsUpdate = true;
      debugLog('Line', 'Points updated', {
        points: newPoints.map(p => p.toArray())
      });
    });

    onBeforeUnmount(() => {
      debugLog('Line', 'Removing from scene', {
        points: props.points.map(p => p.toArray())
      });
      if (parent) {
        parent.remove(line);
      } else if (scene) {
        scene.remove(line);
      }
      geometry.dispose();
      material.dispose();
    });

    return { line };
  }
});

// Html component for labels
export const Html = defineComponent({
  name: 'Html',
  props: {
    position: {
      type: Object as () => THREE.Vector3,
      required: true
    },
    occlude: {
      type: Boolean,
      default: true
    },
    center: {
      type: Boolean,
      default: true
    },
    sprite: {
      type: Boolean,
      default: false
    },
    style: {
      type: Object as () => Record<string, string>,
      default: () => ({})
    }
  },
  setup(props, { slots }) {
    const container = ref<HTMLDivElement | null>(null);

    onMounted(() => {
      if (container.value) {
        const pos = props.position;
        container.value.style.transform = `translate3d(${pos.x}px, ${pos.y}px, ${pos.z}px)`;
        Object.assign(container.value.style, props.style);
        
        if (props.center) {
          container.value.style.transform += ' translate(-50%, -50%)';
        }

        debugLog('Html', 'Mounted', {
          position: [pos.x, pos.y, pos.z],
          center: props.center,
          occlude: props.occlude,
          sprite: props.sprite
        });
      }
    });

    watch(() => props.position, (newPos) => {
      if (container.value) {
        container.value.style.transform = `translate3d(${newPos.x}px, ${newPos.y}px, ${newPos.z}px)`;
        if (props.center) {
          container.value.style.transform += ' translate(-50%, -50%)';
        }
        debugLog('Html', 'Position updated', {
          position: [newPos.x, newPos.y, newPos.z]
        });
      }
    });

    return () => h('div', {
      ref: container,
      class: 'html-overlay',
      style: {
        position: 'absolute',
        pointerEvents: 'none',
        ...props.style
      }
    }, slots.default?.());
  }
});

// Export all components
export default {
  Scene,
  Mesh,
  SphereGeometry,
  MeshStandardMaterial,
  Line,
  Html
};
