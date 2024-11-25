/// <reference types="vite/client" />
/// <reference types="vue/dist/vue.d.ts" />

declare module 'vue' {
  export interface GlobalComponents {
    RouterLink: typeof import('vue-router')['RouterLink']
    RouterView: typeof import('vue-router')['RouterView']
  }

  export {
    defineComponent,
    ref,
    reactive,
    computed,
    watch,
    onMounted,
    onBeforeUnmount,
    shallowRef,
    type PropType,
    type Ref,
    type ComputedRef,
    type ComponentPublicInstance
  } from 'vue'
}

// Strict type checking for store state
declare module 'pinia' {
  export interface PiniaCustomProperties {
    // Add any custom properties here
  }
}

// WebGL context
interface WebGL2RenderingContext {
  readonly COMPUTE_SHADER: number;
  readonly SHADER_STORAGE_BUFFER: number;
  readonly SHADER_STORAGE_BLOCK: number;
  readonly VERTEX_ATTRIB_ARRAY_BARRIER_BIT: number;
  readonly ELEMENT_ARRAY_BARRIER_BIT: number;
  readonly UNIFORM_BARRIER_BIT: number;
  readonly TEXTURE_FETCH_BARRIER_BIT: number;
  readonly SHADER_IMAGE_ACCESS_BARRIER_BIT: number;
  readonly COMMAND_BARRIER_BIT: number;
  readonly PIXEL_BUFFER_BARRIER_BIT: number;
  readonly TEXTURE_UPDATE_BARRIER_BIT: number;
  readonly BUFFER_UPDATE_BARRIER_BIT: number;
  readonly FRAMEBUFFER_BARRIER_BIT: number;
  readonly TRANSFORM_FEEDBACK_BARRIER_BIT: number;
  readonly ATOMIC_COUNTER_BARRIER_BIT: number;
  readonly SHADER_STORAGE_BARRIER_BIT: number;
  readonly ALL_BARRIER_BITS: number;
  readonly MAX_COMPUTE_WORK_GROUP_COUNT: number[];
  readonly MAX_COMPUTE_WORK_GROUP_SIZE: number[];
  readonly MAX_COMPUTE_WORK_GROUP_INVOCATIONS: number;
  readonly MAX_COMPUTE_UNIFORM_BLOCKS: number;
  readonly MAX_COMPUTE_TEXTURE_IMAGE_UNITS: number;
  readonly MAX_COMPUTE_IMAGE_UNIFORMS: number;
  readonly MAX_COMPUTE_SHARED_MEMORY_SIZE: number;
  readonly MAX_COMPUTE_UNIFORM_COMPONENTS: number;
  readonly MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS: number;
  readonly MAX_COMPUTE_ATOMIC_COUNTERS: number;
  readonly MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS: number;
  readonly COMPUTE_WORK_GROUP_SIZE: number;
  readonly DISPATCH_INDIRECT_BUFFER: number;
  readonly DISPATCH_INDIRECT_BUFFER_BINDING: number;
  readonly COMPUTE_SHADER_BIT: number;
}

// Extend Window interface
declare interface Window {
  webkitAudioContext: typeof AudioContext;
  webkitOfflineAudioContext: typeof OfflineAudioContext;
  WebGL2RenderingContext: WebGL2RenderingContext;
}

// Extend HTMLCanvasElement
declare interface HTMLCanvasElement {
  getContext(contextId: "webgl2", contextAttributes?: WebGLContextAttributes): WebGL2RenderingContext | null;
}
