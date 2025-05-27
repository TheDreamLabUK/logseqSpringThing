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
  const [forceUpdate, setForceUpdate] = useState(0) // Force re-render on settings change

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

    // Subscribe to viewport updates from settings store
    // We'll use a different approach - subscribe to the whole store and check for changes
    const unsubscribeViewport = useSettingsStore.subscribe((state, prevState) => {
      // Check if any visualization settings changed
      const visualizationChanged = state.settings?.visualisation !== prevState.settings?.visualisation
      const xrChanged = state.settings?.xr !== prevState.settings?.xr
      const debugChanged = state.settings?.system?.debug !== prevState.settings?.system?.debug
      
      if (visualizationChanged || xrChanged || debugChanged) {
        logger.debug('GraphManager: Detected settings change, forcing update')
        setForceUpdate(prev => prev + 1)
      }
    })

    return () => {
      unsubscribeData()
      unsubscribePositions()
      unsubscribeViewport()
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
    const nodeSize = nodeSettings?.nodeSize || 0.01; // Default if not loaded

    // Log the nodeSize being used
    if (debugState.isEnabled()) { // Only log if debug mode is on
        logger.debug('GraphManager useFrame - nodeSize:', nodeSize);
    }

    let needsUpdate = false;
    graphData.nodes.forEach((node, index) => {
      const pos = node.position;
      if (pos && (pos.x !== 0 || pos.y !== 0 || pos.z !== 0)) {
        // Use nodeSize directly as the scale
        const scale = nodeSize / BASE_SPHERE_RADIUS;
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
    const labelSettings = settings?.visualisation?.labels || {
      enabled: true,
      desktopFontSize: 0.1,
      textColor: '#000000',
      textOutlineColor: '#ffffff',
      textOutlineWidth: 0.01,
      textPadding: 0.3,
      textResolution: 32,
      billboardMode: 'camera'
    };

    const isEnabled = typeof labelSettings === 'object' && labelSettings !== null && 'enabled' in labelSettings ? labelSettings.enabled : true;
    if (!isEnabled) return null;

    const mainLabelFontSize = labelSettings.desktopFontSize || 0.1;
    const metadataFontSize = mainLabelFontSize * 0.7; // Smaller font for metadata
    const lineSpacing = mainLabelFontSize * 0.15; // Space between main label and metadata

    return (
      <group>
        {graphData.nodes.map(node => {
          if (!node.position || !node.label) return null;

          // Construct metadata string
          let metadataString = '';
          if (node.metadata) {
            const fileSize = node.metadata.fileSize; // Already a string from server
            const hyperlinkCount = node.metadata.hyperlinkCount; // Already a string

            if (fileSize) {
              const sizeInKB = parseInt(fileSize, 10) / 1024;
              metadataString += `${sizeInKB.toFixed(1)} KB`;
            }
            if (hyperlinkCount) {
              if (metadataString) metadataString += ' | ';
              metadataString += `${hyperlinkCount} links`;
            }
            // Could add lastModified here too, but might be too much info
            // const lastModified = node.metadata.lastModified;
            // if (lastModified) {
            //   if (metadataString) metadataString += ' | ';
            //   metadataString += `Mod: ${new Date(lastModified).toLocaleDateString()}`;
            // }
          }

          return (
            <Billboard
              key={node.id}
              // Position the billboard slightly above the node center to accommodate two lines of text
              position={[
                node.position.x,
                node.position.y + (labelSettings.textPadding || 0.3) + (metadataString ? mainLabelFontSize / 2 + lineSpacing / 2 : 0),
                node.position.z
              ]}
              follow={labelSettings.billboardMode === 'camera'}
            >
              {/* Main Label (Filename) */}
              <Text
                fontSize={mainLabelFontSize}
                color={labelSettings.textColor || '#000000'}
                anchorX="center"
                anchorY="middle" // Anchor to middle for the main label
                outlineWidth={labelSettings.textOutlineWidth || 0.01}
                outlineColor={labelSettings.textOutlineColor || '#ffffff'}
                outlineOpacity={1.0}
                renderOrder={10}
                material-depthTest={false}
                maxWidth={labelSettings.textResolution || 32}
              >
                {node.label}
              </Text>

              {/* Metadata String (File Size, Links) - rendered below the main label */}
              {metadataString && (
                <Text
                  fontSize={metadataFontSize}
                  color={labelSettings.textColor ? new THREE.Color(labelSettings.textColor).multiplyScalar(0.8).getStyle() : '#333333'} // Slightly dimmer
                  anchorX="center"
                  anchorY="top" // Anchor to top, so it sits below the main label
                  position={[0, -mainLabelFontSize / 2 - lineSpacing, 0]} // Position it below the main label
                  outlineWidth={(labelSettings.textOutlineWidth || 0.01) * 0.7}
                  outlineColor={labelSettings.textOutlineColor || '#ffffff'}
                  outlineOpacity={0.8}
                  renderOrder={10}
                  material-depthTest={false}
                  maxWidth={(labelSettings.textResolution || 32) * 1.5} // Allow metadata to be a bit wider
                >
                  {metadataString}
                </Text>
              )}
            </Billboard>
          );
        })}
      </group>
    );
  };

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


export default GraphManager
