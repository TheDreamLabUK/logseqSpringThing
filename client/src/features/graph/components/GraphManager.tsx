import React, { useRef, useEffect, useState, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import { Line } from '@react-three/drei/core/Line'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { createLogger } from '../../../utils/logger'
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
  const meshRef = useRef(null)
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [nodesAreAtOrigin, setNodesAreAtOrigin] = useState(false)
  const settings = useSettingsStore(state => state.settings)

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
  const updateNodePositions = (positions: Float32Array) => {
    const mesh = meshRef.current
    if (!mesh) return

    const nodeCount = positions.length / 4
    mesh.count = nodeCount

    for (let i = 0; i < nodeCount; i++) {
      const x = positions[i * 4]
      const y = positions[i * 4 + 1]
      const z = positions[i * 4 + 2]

      if (isNaN(x) || isNaN(y) || isNaN(z)) {
        mesh.setMatrixAt(i, new Float32Array([
          0,0,0,0,
          0,0,0,0,
          0,0,0,0,
          0,0,0,1
        ]))
      } else {
        mesh.setMatrixAt(i, new Float32Array([
          0.2,0,0,0,
          0,0.2,0,0,
          0,0,0.2,0,
          x,y,z,1
        ]))
      }
    }

    mesh.instanceMatrix.needsUpdate = true
  }

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

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[null, null, graphData.nodes.length]}
        frustumCulled={false}
      >
        <sphereGeometry args={[0.2, 8, 8]} />
        <meshStandardMaterial
          // vertexColors // Removed, not a valid prop for meshStandardMaterial
          metalness={0.1}
          roughness={0.5}
          emissive="#222222"
        />
      </instancedMesh>

      {edgePoints.length > 0 && (
        <Line
          points={edgePoints}
          color="white"
          lineWidth={0.5}
          transparent
          opacity={0.3}
        />
      )}
    </>
  )
}

export default GraphManager