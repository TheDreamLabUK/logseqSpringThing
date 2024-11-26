/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

declare module 'vue-threejs' {
  import { Plugin } from 'vue'
  const VueThreejs: Plugin
  export default VueThreejs
}
