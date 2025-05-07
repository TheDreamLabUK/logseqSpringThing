// import * as THREE from 'three';
// import React from 'react';
// import { ReactThreeFiber, Object3DNode } from '@react-three/fiber'; // Import R3F namespace for types
//
// declare module '@react-three/fiber' {
//   // Core React Three Fiber hooks and components
//   export function Canvas(props: any): JSX.Element;
//   export function useThree(): {
//     gl: THREE.WebGLRenderer;
//     scene: THREE.Scene;
//     camera: THREE.Camera;
//     size: { width: number; height: number };
//     viewport: { width: number; height: number; factor: number };
//     raycaster: THREE.Raycaster;
//     mouse: THREE.Vector2;
//     clock: THREE.Clock;
//     // Add other context properties as needed
//   };
//   export function useFrame(callback: (state: any, delta: number) => void, renderPriority?: number): void;
//
//   // Extend mesh props for better TypeScript integration with jsx-runtime
//   export interface MeshProps {
//     color?: string | number | THREE.Color;
//     wireframe?: boolean;
//     transparent?: boolean;
//     opacity?: number;
//     side?: typeof THREE.FrontSide | typeof THREE.BackSide | typeof THREE.DoubleSide;
//     emissive?: string | number | THREE.Color;
//     emissiveIntensity?: number;
//     depthWrite?: boolean;
//     roughness?: number;
//     thickness?: number;
//     transmission?: number;
//     distortion?: number;
//     temporalDistortion?: number;
//     clearcoat?: number;
//     attenuationDistance?: number;
//     attenuationColor?: string | number | THREE.Color;
//     ref?: React.Ref<any>;
//   }
//
//   export interface ExtendedColors<T> {
//     color?: string | number | THREE.Color;
//     emissive?: string | number | THREE.Color;
//     // Add other color properties as needed
//   }
//
// }
//
// // Define MeshTransmissionMaterial props
// declare module '@react-three/drei' {
//   export interface MeshTransmissionMaterialProps {
//     transmissionSampler?: boolean;
//     backside?: boolean;
//     samples?: number;
//     resolution?: number;
//     transmission?: number;
//     roughness?: number;
//     thickness?: number;
//     ior?: number;
//     chromaticAberration?: number;
//     anisotropy?: number;
//     distortion?: number;
//     distortionScale?: number;
//     temporalDistortion?: number;
//     clearcoat?: number;
//     attenuationDistance?: number;
//     attenuationColor?: string | number | THREE.Color;
//     color?: string | number | THREE.Color;
//     bg?: string | number | THREE.Color;
//   }
//
//   export type MeshTransmissionMaterialType = THREE.Material & {
//     // Add specific props of the material implementation if needed
//   };
// }
//
// // Augment the global JSX namespace
// declare global {
//   namespace JSX {
//     interface IntrinsicElements {
//       // built-in three.js lights
//       ambientLight:    Object3DNode<THREE.AmbientLight,    typeof THREE.AmbientLight>
//       directionalLight: Object3DNode<THREE.DirectionalLight, typeof THREE.DirectionalLight>
//       pointLight:      Object3DNode<THREE.PointLight,      typeof THREE.PointLight>
//       // helper / misc
//       axesHelper:      Object3DNode<THREE.AxesHelper,      typeof THREE.AxesHelper>
//       color:           Object3DNode<THREE.Color,           typeof THREE.Color>
//       // Elements used in GraphManager.tsx & XR components
//       group:           Object3DNode<THREE.Group,           typeof THREE.Group>
//       instancedMesh:   Object3DNode<THREE.InstancedMesh,   typeof THREE.InstancedMesh>
//       sphereGeometry:  Object3DNode<THREE.SphereGeometry,  typeof THREE.SphereGeometry>
//       meshStandardMaterial: Object3DNode<THREE.MeshStandardMaterial, typeof THREE.MeshStandardMaterial>
//       // Added based on XR component errors
//       mesh:            Object3DNode<THREE.Mesh,            typeof THREE.Mesh>
//       planeGeometry:   Object3DNode<THREE.PlaneGeometry,   typeof THREE.PlaneGeometry>
//       // ...add any others you need (e.g. GridHelper, etc.)
//     }
//   }
// }
