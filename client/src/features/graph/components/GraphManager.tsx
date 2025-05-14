import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
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
    if (meshRef.current) {
      const count = graphData.nodes.length;
      const mesh = meshRef.current;
      mesh.count = count; // Set the count

      if (count > 0) {
        // Check if matrices need initialization (e.g., if they are identity)
        // This avoids re-initializing if positions are already set by useFrame
        let needsInitialization = false;
        const identityMatrix = new THREE.Matrix4(); // Re-use for comparison
        for (let i = 0; i < count; i++) {
          const currentMatrix = new THREE.Matrix4();
          // Ensure mesh has enough allocated matrices before calling getMatrixAt
          if (i < mesh.instanceMatrix.array.length / 16) { // 16 floats per matrix
            mesh.getMatrixAt(i, currentMatrix);
            if (currentMatrix.equals(identityMatrix)) {
              needsInitialization = true;
              break;
            }
          } else {
            // If count increased beyond allocated, it needs initialization
            needsInitialization = true;
            break;
          }
        }

        if (needsInitialization) {
          for (let i = 0; i < count; i++) {
            // Set to identity or a default non-zero position if appropriate
            mesh.setMatrixAt(i, tempMatrix.identity());
          }
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (debugState.isEnabled()) {
        logger.debug(`InstancedMesh count updated to: ${count}`);
      }
    }
  }, [graphData.nodes.length, tempMatrix]);

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
  const updateNodePositions = useCallback((positions: Float32Array) => {
    // This function is called by GraphDataManager when WebSocket binary data arrives.
    // GraphDataManager is responsible for updating the central 'graphData' state.
    // This component (GraphManager) re-renders when 'graphData' (from useState) changes.
    // The useFrame hook then uses the updated 'graphData' to set instance matrices.
    // Therefore, this callback doesn't need to directly manipulate meshRef.current.
    if (debugState.isEnabled()) {
      const sample = positions.slice(0, Math.min(12, positions.length)); // Log first few nodes
      logger.debug('GraphManager received raw position update data (sample):', sample);
    }
  }, []); // No dependencies needed if it's just logging or relying on external state updates.

  // Constants for file size normalization
  const MIN_LOG_FILE_SIZE_ESTIMATE = Math.log10(100 + 1); // Approx 2, for 100 bytes
  const MAX_LOG_FILE_SIZE_ESTIMATE = Math.log10(5 * 1024 * 1024 + 1); // Approx 6.7, for 5MB
  const BASE_SPHERE_RADIUS = 0.5;

  useFrame(() => {
    if (!meshRef.current) return;

    const nodeSettings = settings?.visualisation?.nodes;
    const settingsSizeRange = nodeSettings?.sizeRange || [0.5, 1.5]; // Default if not loaded

    // Log the settingsSizeRange being used
    if (debugState.isEnabled()) { // Only log if debug mode is on
        logger.debug('GraphManager useFrame - settingsSizeRange:', settingsSizeRange);
    }

    let needsUpdate = false;
    graphData.nodes.forEach((node, index) => {
      const pos = node.position;
      if (pos && (pos.x !== 0 || pos.y !== 0 || pos.z !== 0)) {
        const scale = calculateNodeScale(
          node,
          settingsSizeRange,
          MIN_LOG_FILE_SIZE_ESTIMATE,
          MAX_LOG_FILE_SIZE_ESTIMATE,
          BASE_SPHERE_RADIUS
        );
        updateInstanceMatrix(index, pos.x, pos.y, pos.z, scale);
        needsUpdate = true;
      }
    });

    if (needsUpdate) {
      meshRef.current.instanceMatrix.needsUpdate = true;
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

// Helper function to calculate node scale based on metadata and settings
const calculateNodeScale = (
  node: GraphNode,
  settingsSizeRange: [number, number],
  minLogFileSizeEstimate: number,
  maxLogFileSizeEstimate: number,
  baseSphereRadius: number
): number => {
  const targetMinRadius = settingsSizeRange[0];
  const targetMaxRadius = settingsSizeRange[1];

  let normalizedValue = 0; // Default to min size if no metadata

  if (node.metadata?.fileSize) {
    const fileSize = parseInt(node.metadata.fileSize, 10);
    if (!isNaN(fileSize) && fileSize > 0) {
      const logFileSize = Math.log10(fileSize + 1);
      if (maxLogFileSizeEstimate > minLogFileSizeEstimate) {
        normalizedValue = (logFileSize - minLogFileSizeEstimate) / (maxLogFileSizeEstimate - minLogFileSizeEstimate);
      }
    }
  } else if (node.metadata?.size) {
    // Fallback for metadata.size, assuming it's a direct value that needs normalization
    // This part might need adjustment based on typical range of node.metadata.size
    const numericSize = parseFloat(node.metadata.size);
    if (!isNaN(numericSize)) {
      // Example: if metadata.size is 0-100, this normalizes it.
      // Adjust 100 if the expected range is different.
      const assumedMaxMetadataSize = 100;
      normalizedValue = numericSize / assumedMaxMetadataSize;
    }
  }

  // Clamp normalizedValue to be strictly between 0 and 1
  normalizedValue = Math.max(0, Math.min(normalizedValue, 1));

  // Determine final radius based on normalized value and settings range
  const finalRadius = targetMinRadius + normalizedValue * (targetMaxRadius - targetMinRadius);

  // Calculate scale factor based on the base sphere radius
  const scaleFactor = finalRadius / baseSphereRadius;

  return scaleFactor;
};

export default GraphManager
