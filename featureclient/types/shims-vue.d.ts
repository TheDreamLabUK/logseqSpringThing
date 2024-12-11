declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

declare module 'vue-threejs' {
  import type { Plugin } from 'vue'
  const VueThreejs: Plugin
  export default VueThreejs
}

declare module 'three/*' {
  import * as THREE from 'three'
  export * from 'three'
}
