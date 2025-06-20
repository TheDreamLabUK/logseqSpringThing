import React from 'react'
import { Canvas } from '@react-three/fiber'
import { XR, Controllers, Hands } from '@react-three/xr'
import { Environment, OrbitControls } from '@react-three/drei'
import GraphManager from '../../graph/components/GraphManager'; // Changed to default import
import { useSettingsStore } from '../../../store/settingsStore'; // Corrected path

export const XRScene = () => {
  const settings = useSettingsStore(state => state.settings)
  const xrSettings = settings?.xr

  return (
    <Canvas
      camera={{ position: [0, 1.6, 3], fov: 50 }}
      shadows
    >
      <XR>
        <Controllers />
        <Hands />
        
        {/* AR-specific lighting */}
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
        />
        
        {/* Environment for better AR visualisation */}
        <Environment preset="city" />
        
        {/* Main graph visualisation */}
        <GraphManager />
        
        {/* Optional orbit controls for non-AR mode */}
        {!xrSettings?.enabled && (
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            target={[0, 0, 0]}
          />
        )}
      </XR>
    </Canvas>
  )
} 