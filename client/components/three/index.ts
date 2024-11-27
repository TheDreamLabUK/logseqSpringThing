import { defineComponent, h, onMounted, onBeforeUnmount, watch, ref } from 'vue'
import * as THREE from 'three'

// Define our supported event types
type SupportedEventType = 'click' | 'pointerenter' | 'pointerleave' | 'pointerdown' | 'pointermove' | 'pointerup';

// Extend Three.js event map
interface CustomObject3DEventMap extends THREE.Object3DEventMap {
  click: THREE.Event;
  pointerover: THREE.Event;
  pointerout: THREE.Event;
  pointerdown: THREE.Event;
  pointermove: THREE.Event;
  pointerup: THREE.Event;
}

// Map our event types to Three.js event types
const eventMap: Record<SupportedEventType, keyof CustomObject3DEventMap> = {
  click: 'click',
  pointerenter: 'pointerover',
  pointerleave: 'pointerout',
  pointerdown: 'pointerdown',
  pointermove: 'pointermove',
  pointerup: 'pointerup'
} as const;

// Basic Three.js component wrapper
export const Group = defineComponent({
  name: 'Group',
  setup(_, { slots }) {
    const group = new THREE.Group()
    
    onMounted(() => {
      // Parent will handle adding to scene
    })

    onBeforeUnmount(() => {
      group.clear()
    })

    return () => h('div', slots.default?.())
  }
})

export const Scene = defineComponent({
  name: 'Scene',
  props: {
    background: {
      type: [Number, String],
      default: 0x000000
    }
  },
  setup(props) {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(props.background)

    watch(() => props.background, (newColor) => {
      scene.background = new THREE.Color(newColor)
    })

    onBeforeUnmount(() => {
      scene.clear()
    })

    return { scene }
  }
})

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
  setup(props, { emit }) {
    const mesh = ref<THREE.Mesh | null>(null)

    onMounted(() => {
      if (mesh.value) {
        mesh.value.position.copy(props.position)
        mesh.value.scale.set(props.scale.x, props.scale.y, props.scale.z)

        // Event handlers
        Object.entries(eventMap).forEach(([emitType, threeType]) => {
          const handler = (event: THREE.Event) => {
            emit(emitType as SupportedEventType, event)
          }
          mesh.value?.addEventListener(threeType as keyof THREE.Object3DEventMap, handler)
        })
      }
    })

    watch(() => props.position, (newPos) => {
      if (mesh.value) {
        mesh.value.position.copy(newPos)
      }
    })

    watch(() => props.scale, (newScale) => {
      if (mesh.value) {
        mesh.value.scale.set(newScale.x, newScale.y, newScale.z)
      }
    })

    onBeforeUnmount(() => {
      if (mesh.value) {
        mesh.value.geometry.dispose()
        if (Array.isArray(mesh.value.material)) {
          mesh.value.material.forEach(m => m.dispose())
        } else {
          mesh.value.material.dispose()
        }
      }
    })

    return { mesh }
  }
})

export const SphereGeometry = defineComponent({
  name: 'SphereGeometry',
  props: {
    args: {
      type: Array as unknown as () => [number, number, number],
      default: () => [1, 32, 32] as [number, number, number]
    }
  },
  setup(props) {
    const geometry = new THREE.SphereGeometry(...props.args)

    onBeforeUnmount(() => {
      geometry.dispose()
    })

    return { geometry }
  }
})

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
    })

    watch(() => props.color, (newColor) => {
      material.color.set(newColor)
    })

    watch(() => props.emissive, (newColor) => {
      material.emissive.set(newColor)
    })

    onBeforeUnmount(() => {
      material.dispose()
    })

    return { material }
  }
})

export const Line = defineComponent({
  name: 'Line',
  props: {
    points: {
      type: Array as unknown as () => [THREE.Vector3, THREE.Vector3],
      required: true,
      validator: (value: unknown) => {
        if (!Array.isArray(value) || value.length !== 2) return false
        return value.every(v => v instanceof THREE.Vector3)
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
    const geometry = new THREE.BufferGeometry().setFromPoints(props.points)
    const material = new THREE.LineBasicMaterial({
      color: props.color,
      linewidth: props.linewidth,
      opacity: props.opacity,
      transparent: props.transparent
    })
    const line = new THREE.Line(geometry, material)

    watch(() => props.points, (newPoints) => {
      geometry.setFromPoints(newPoints)
      geometry.attributes.position.needsUpdate = true
    })

    onBeforeUnmount(() => {
      geometry.dispose()
      material.dispose()
    })

    return { line }
  }
})

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
    const container = ref<HTMLDivElement | null>(null)

    onMounted(() => {
      if (container.value) {
        // Position the HTML element
        const pos = props.position
        container.value.style.transform = `translate3d(${pos.x}px, ${pos.y}px, ${pos.z}px)`
        
        // Apply styles
        Object.assign(container.value.style, props.style)
        
        if (props.center) {
          container.value.style.transform += ' translate(-50%, -50%)'
        }
      }
    })

    watch(() => props.position, (newPos) => {
      if (container.value) {
        container.value.style.transform = `translate3d(${newPos.x}px, ${newPos.y}px, ${newPos.z}px)`
        if (props.center) {
          container.value.style.transform += ' translate(-50%, -50%)'
        }
      }
    })

    return () => h('div', {
      ref: container,
      class: 'html-overlay',
      style: {
        position: 'absolute',
        pointerEvents: 'none',
        ...props.style
      }
    }, slots.default?.())
  }
})

// Export all components
export default {
  Group,
  Scene,
  Mesh,
  SphereGeometry,
  MeshStandardMaterial,
  Line,
  Html
}
