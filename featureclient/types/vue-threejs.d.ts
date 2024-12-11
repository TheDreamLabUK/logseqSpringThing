declare module 'vue-threejs' {
  import { Plugin, Component } from 'vue'
  import { Scene, PerspectiveCamera, WebGLRenderer, Vector3, Color } from 'three'

  interface RendererProps {
    antialias?: boolean;
    xr?: boolean;
    size: {
      w: number;
      h: number;
    };
  }

  interface SceneProps {
    ref?: string;
  }

  interface GroupProps {
    ref?: string;
    position?: { x: number; y: number; z: number };
    rotation?: { x: number; y: number; z: number };
    scale?: { x: number; y: number; z: number };
  }

  interface CameraProps {
    position?: {
      x: number;
      y: number;
      z: number;
    };
    fov?: number;
    aspect?: number;
    near?: number;
    far?: number;
  }

  interface LightProps {
    intensity?: number;
    position?: {
      x: number;
      y: number;
      z: number;
    };
    color?: number | string;
    castShadow?: boolean;
  }

  interface HemisphereLightProps extends LightProps {
    skyColor?: number | string;
    groundColor?: number | string;
  }

  interface MeshProps {
    position?: { x: number; y: number; z: number };
    rotation?: { x: number; y: number; z: number };
    scale?: { x: number; y: number; z: number } | number;
  }

  interface GeometryProps {
    args?: any[];
  }

  interface MaterialProps {
    color?: number | string;
    metalness?: number;
    roughness?: number;
    opacity?: number;
    transparent?: boolean;
  }

  interface LineProps {
    points: Vector3[];
    color?: number | string;
    linewidth?: number;
    opacity?: number;
    transparent?: boolean;
  }

  interface HtmlProps {
    position: Vector3;
    occlude?: boolean;
    center?: boolean;
    sprite?: boolean;
  }

  export const Renderer: Component<RendererProps>
  export const Scene: Component<SceneProps>
  export const Group: Component<GroupProps>
  export const Camera: Component<CameraProps>
  export const AmbientLight: Component<LightProps>
  export const DirectionalLight: Component<LightProps>
  export const HemisphereLight: Component<HemisphereLightProps>
  export const Mesh: Component<MeshProps>
  export const SphereGeometry: Component<GeometryProps>
  export const MeshStandardMaterial: Component<MaterialProps>
  export const Line: Component<LineProps>
  export const Html: Component<HtmlProps>

  const VueThreejs: Plugin
  export default VueThreejs
}
