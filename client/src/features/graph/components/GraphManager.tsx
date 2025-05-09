import React, { useRef, useEffect, useState, useMemo } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei/core/Line'
// Assuming Text and Billboard are still directly available, if not adjust path later
import { Text, Billboard } from '@react-three/drei'
// Use namespace import for THREE to access constructors
import * as THREE from 'three'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { createLogger, createErrorMetadata } from '../../../utils/logger'
import { debugState } from '../../../utils/debugState'
import { useSettingsStore } from '../../../store/settingsStore'

const logger = createLogger('GraphManager')

// Function to get random position if node is at origin
const getPositionForNode = (node: GraphNode, index: number): [number, number, number] => {
  if (!node.position ||
      (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)) {
    // All nodes are at (0,0,0), so generate a random position in a sphere
    const radius = 10
    const phi = Math.acos(2 * Math.random() - 1)
    const theta = Math.random() * Math.PI * 2

    const x = radius * Math.sin(phi) * Math.cos(theta)
    const y = radius * Math.sin(phi) * Math.sin(theta)
    const z = radius * Math.cos(phi)

    // Update the original node position so edges will work
    if (node.position) {
      node.position.x = x
      node.position.y = y
      node.position.z = z
    } else {
      node.position = { x, y, z }
    }

    return [x, y, z]
  }

  return [node.position.x, node.position.y, node.position.z]
}

const GraphManager = () => {
  const meshRef = useRef<THREE.InstancedMesh>(null) // Initialize with null, use THREE namespace
  // Use useMemo for stable object references across renders
  const tempMatrix = useMemo(() => new THREE.Matrix4(), [])
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempScale = useMemo(() => new THREE.Vector3(), [])
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [nodesAreAtOrigin, setNodesAreAtOrigin] = useState(false)
  const settings = useSettingsStore(state => state.settings)

  useEffect(() => {
    if (!meshRef.current) return
    
    // Initialize all matrices to prevent undefined states
    const mesh = meshRef.current
    const count = graphData.nodes.length
    mesh.count = count
    
    // Initialize all instances with identity matrix
    for (let i = 0; i < count; i++) {
      mesh.setMatrixAt(i, tempMatrix.identity())
    }
    mesh.instanceMatrix.needsUpdate = true
    
    console.debug(`Initialized ${count} instances`)
  }, [graphData.nodes.length])

  // Separate matrix update function for better performance
  const updateInstanceMatrix = (
    index: number,
    x: number,
    y: number,
    z: number,
    scale: number
  ) => {
    if (!meshRef.current) return

    tempPosition.set(x, y, z)
    tempScale.set(scale, scale, scale)
    
    tempMatrix.makeScale(scale, scale, scale)
    tempMatrix.setPosition(tempPosition)
    
    meshRef.current.setMatrixAt(index, tempMatrix)
  }

  // Subscribe to graph data changes
  useEffect(() => {
    const handleGraphDataChange = (newData: GraphData) => {
      setGraphData(newData)

      // Check if nodes are all at origin
      const allAtOrigin = newData.nodes.every(node =>
        !node.position || (node.position.x === 0 && node.position.y === 0 && node.position.z === 0)
      )
      setNodesAreAtOrigin(allAtOrigin)
    }

    // Initial data load
    const initialData = graphDataManager.getGraphData()
    handleGraphDataChange(initialData)

    // Subscribe to updates
    const unsubscribeData = graphDataManager.onGraphDataChange(handleGraphDataChange)
    const unsubscribePositions = graphDataManager.onPositionUpdate((positions) => {
      updateNodePositions(positions)
    })

    return () => {
      unsubscribeData()
      unsubscribePositions()
    }
  }, [])

  // Update node positions from binary data
  // Update node positions - Modified to NOT directly update mesh matrices from WebSocket data
  const updateNodePositions = (positions: Float32Array) => {
    // This function is called when position updates arrive via WebSocket.
    // Based on feedback, this data might not be the absolute coordinates.
    // We will rely on the useFrame loop to update matrices from the graphData state.
    // We might need to use this data differently later (e.g., updating state or metadata).

    const mesh = meshRef.current
    if (!mesh) return

    // We might still need to update the count based on WebSocket data if nodes appear/disappear
    const nodeCount = positions.length / 4
    if (mesh.count !== nodeCount) {
        mesh.count = nodeCount;
        // Mark matrix as needing update if count changes, although positions are set in useFrame
        mesh.instanceMatrix.needsUpdate = true;
    }

    // TODO: Determine the correct way to use the 'positions' data.
    // For now, we log it if debugging is enabled.
    if (debugState.isEnabled()) {
      // Log only a small sample to avoid flooding console
      const sample = positions.slice(0, 12); // Log first 3 nodes' data
      logger.debug('Received position update data (sample):', sample);
    }

    // Do NOT update mesh matrices here. Let useFrame handle it based on graphData state.
  }

  useFrame(() => {
    if (!meshRef.current) return

    let needsUpdate = false
    graphData.nodes.forEach((node, index) => {
      const pos = node.position // Access position directly, assuming it exists on GraphNode type
      if (pos && (pos.x !== 0 || pos.y !== 0 || pos.z !== 0)) {
        const scale = calculateNodeScale(node) // Implement this based on your needs
        updateInstanceMatrix(index, pos.x, pos.y, pos.z, scale)
        needsUpdate = true
      }
    })

    if (needsUpdate) {
      meshRef.current.instanceMatrix.needsUpdate = true
    }
  })

  // Memoize edge points
  const edgePoints = useMemo(() => {
    if (!graphData.nodes || !graphData.edges) return []

    const points: [number, number, number][] = []
    const { nodes, edges } = graphData

    edges.forEach(edge => {
      if (edge.source && edge.target) {
        const sourceNode = nodes.find(n => n.id === edge.source)
        const targetNode = nodes.find(n => n.id === edge.target)
        if (sourceNode?.position && targetNode?.position) {
          if (nodesAreAtOrigin) {
            points.push(
              getPositionForNode(sourceNode, nodes.indexOf(sourceNode)),
              getPositionForNode(targetNode, nodes.indexOf(targetNode))
            )
          } else {
            points.push(
              [sourceNode.position.x, sourceNode.position.y, sourceNode.position.z],
              [targetNode.position.x, targetNode.position.y, targetNode.position.z]
            )
          }
        }
      }
    })
    return points
  }, [graphData.nodes, graphData.edges, nodesAreAtOrigin])

  // Node labels component using settings from YAML
  const NodeLabels = () => {
    // Get label settings from the settings store (in camelCase)
    const labelSettings = settings?.visualisation?.labels || {
      enabled: true,
      desktopFontSize: 0.1, // Fallback to a small size if not specified
      textColor: '#000000',
      textOutlineColor: '#ffffff',
      textOutlineWidth: 0.01,
      textPadding: 0.3,
      textResolution: 32,
      billboardMode: 'camera'
    }

    // Don't render if labels are disabled
    // Type guard to safely access 'enabled' property
    const isEnabled = typeof labelSettings === 'object' && labelSettings !== null && 'enabled' in labelSettings ? labelSettings.enabled : true; // Default to true if structure is unexpected
    if (!isEnabled) return null

    // Use the desktopFontSize (camelCase) from settings
    // The settings are converted from snake_case to camelCase when loaded
    const fontSize = labelSettings.desktopFontSize || 0.1

    return (
      <group>
        {graphData.nodes.map(node => {
          // Skip nodes without position or label
          if (!node.position || !node.label) return null

          // Use the font size directly from settings without any scaling

          return (
            <Billboard
              key={node.id}
              position={[node.position.x, node.position.y + (labelSettings.textPadding || 0.3), node.position.z]} // Use textPadding from settings
              follow={labelSettings.billboardMode === 'camera'} // Use billboardMode from settings
            >
              <Text
                fontSize={fontSize}
                color={labelSettings.textColor || '#000000'}
                anchorX="center"
                anchorY="middle"
                outlineWidth={labelSettings.textOutlineWidth || 0.01}
                outlineColor={labelSettings.textOutlineColor || '#ffffff'}
                outlineOpacity={1.0} // Full opacity for outline
                renderOrder={10}
                material-depthTest={false}
                maxWidth={labelSettings.textResolution || 32} // Use textResolution for max width
              >
                {node.label}
              </Text>
            </Billboard>
          )
        })}
      </group>
    )
  }

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[null, null, graphData.nodes.length]}
        frustumCulled={false}
      >
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshStandardMaterial
          color={settings?.visualisation?.nodes?.baseColor || "#ffffff"}
          emissive={settings?.visualisation?.nodes?.baseColor || "#00ffff"}
          emissiveIntensity={0.8}
          metalness={settings?.visualisation?.nodes?.metalness || 0.2}
          roughness={settings?.visualisation?.nodes?.roughness || 0.3}
          opacity={settings?.visualisation?.nodes?.opacity || 1.0}
          transparent={true}
          toneMapped={false} // Important for bloom effect
        />
      </instancedMesh>

      {edgePoints.length > 0 && (
        <Line
          points={edgePoints}
          color={settings?.visualisation?.edges?.color || "#00ffff"}
          lineWidth={settings?.visualisation?.edges?.baseWidth || 1.0}
          transparent
          opacity={settings?.visualisation?.edges?.opacity || 0.6}
          toneMapped={false} // Important for bloom effect
        />
      )}

      <NodeLabels />
    </>
  )
}

// Helper function to calculate node scale based on metadata
const calculateNodeScale = (node: any) => {
  let scale = 1.0
  
  if (node.metadata?.fileSize) {
    // Logarithmic scale based on file size
    scale = Math.log10(parseInt(node.metadata.fileSize) + 1) * 0.1 + 0.5
  } else if (node.metadata?.size) {
    scale = parseFloat(node.metadata.size) / 100
  }
  
  // Clamp scale to reasonable values
  return Math.max(0.2, Math.min(scale, 2.0))
}

export default GraphManager
