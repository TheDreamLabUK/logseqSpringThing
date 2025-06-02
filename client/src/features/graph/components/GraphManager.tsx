import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react'
import { useThree, useFrame, ThreeEvent } from '@react-three/fiber'
import { Line } from '@react-three/drei/core/Line'
// Assuming Text and Billboard are still directly available, if not adjust path later
import { Text, Billboard } from '@react-three/drei' // OrbitControls removed
// Use namespace import for THREE to access constructors
import * as THREE from 'three'
import { graphDataManager, type GraphData, type Node as GraphNode } from '../managers/graphDataManager'
import { createLogger, createErrorMetadata } from '../../../utils/logger'
import { debugState } from '../../../utils/debugState'
import { useSettingsStore } from '../../../store/settingsStore'
import { BinaryNodeData, createBinaryNodeData } from '../../../types/binaryProtocol'

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

// Define props for GraphManager
interface GraphManagerProps {
  onNodeDragStateChange: (isDragging: boolean) => void;
}

const GraphManager: React.FC<GraphManagerProps> = ({ onNodeDragStateChange }) => { // Accept prop
  const meshRef = useRef<THREE.InstancedMesh>(null) // Initialize with null, use THREE namespace
  // REMOVE: const orbitControlsRef = useRef<any>(null);
  
  // Use useMemo for stable object references across renders
  const tempMatrix = useMemo(() => new THREE.Matrix4(), [])
  const tempPosition = useMemo(() => new THREE.Vector3(), [])
  const tempScale = useMemo(() => new THREE.Vector3(), [])
  const tempQuaternion = useMemo(() => new THREE.Quaternion(), [])
  const screenPosition = useMemo(() => new THREE.Vector2(), [])
  
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [nodesAreAtOrigin, setNodesAreAtOrigin] = useState(false)
  const settings = useSettingsStore(state => state.settings)
  const [forceUpdate, setForceUpdate] = useState(0) // Force re-render on settings change
  
  // Minimal drag state for UI feedback
  const [dragState, setDragState] = useState<{
    nodeId: string | null;
    instanceId: number | null;
  }>({ nodeId: null, instanceId: null });

  // Performance-optimized drag data using refs (no re-renders)
  const dragDataRef = useRef({
    isDragging: false,
    nodeId: null as string | null,
    instanceId: null as number | null,
    startPosition: new THREE.Vector3(),
    currentPosition: new THREE.Vector3(),
    offset: new THREE.Vector2(),
    lastUpdateTime: 0,
    pendingUpdate: null as BinaryNodeData | null
  });

  const { camera, size } = useThree()

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

  // Optimized drag event handlers
  const clickTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastClickTimeRef = useRef<number>(0);
  const DOUBLE_CLICK_THRESHOLD = 300; // ms

  const slugifyNodeLabel = (label: string): string => {
    return label.toLowerCase().replace(/\s+/g, '%20');
  };

  const handleNodeClick = useCallback((event: ThreeEvent<PointerEvent>) => {
    const instanceId = event.instanceId;
    if (instanceId === undefined) return;

    event.stopPropagation();
    const node = graphData.nodes[instanceId];
    if (!node || !node.label) return;

    const currentTime = Date.now();

    if (clickTimeoutRef.current) {
      clearTimeout(clickTimeoutRef.current);
      clickTimeoutRef.current = null;
    }

    // Double click
    if (currentTime - lastClickTimeRef.current < DOUBLE_CLICK_THRESHOLD) {
      if (debugState.isEnabled()) {
        logger.debug(`Double-clicked node ${node.id} (instance ${instanceId})`);
      }
      // Initiate drag on double click
      const pointer = event.pointer;
      dragDataRef.current = {
        isDragging: true,
        nodeId: node.id,
        instanceId,
        startPosition: new THREE.Vector3(node.position.x, node.position.y, node.position.z),
        currentPosition: new THREE.Vector3(node.position.x, node.position.y, node.position.z),
        offset: new THREE.Vector2(pointer.x, pointer.y),
        lastUpdateTime: 0,
        pendingUpdate: null
      };
      onNodeDragStateChange(true);
      setDragState({ nodeId: node.id, instanceId });
      lastClickTimeRef.current = 0; // Reset for next double click
    } else {
      // Single click (or first click of a potential double click)
      clickTimeoutRef.current = setTimeout(() => {
        if (debugState.isEnabled()) {
          logger.debug(`Single-clicked node ${node.id} (instance ${instanceId})`);
        }
        const slug = slugifyNodeLabel(node.label!);
        const narrativeGoldmineUrl = `https://narrativegoldmine.com//#/page/${slug}`;
        // This assumes the Narrative Goldmine panel is an iframe or a component that listens to URL changes.
        // If it's an iframe, you might target its src. If it's a React component, you might use react-router or a state management solution.
        // For now, let's log it. A more robust solution would involve a shared service or context.
        logger.info(`Updating Narrative Goldmine URL to: ${narrativeGoldmineUrl}`);
        // Example: window.postMessage({ type: 'UPDATE_NARRATIVE_URL', url: narrativeGoldmineUrl }, '*');
        // Or if it's a sibling iframe:
        // const iframe = document.getElementById('narrative-goldmine-iframe') as HTMLIFrameElement;
        // if (iframe) iframe.src = narrativeGoldmineUrl;

        // To actually change the browser's URL (if Narrative Goldmine is part of the same SPA but different route):
        // window.history.pushState({}, '', narrativeGoldmineUrl); // or router.push(...)

        // For now, we'll assume a global event or direct update if possible.
        // This part needs to be integrated with how NarrativeGoldminePanel actually receives its URL.
        // One simple way, if it's an iframe with a known ID:
        const narrativeIframe = document.getElementById('narrative-goldmine-iframe') as HTMLIFrameElement | null;
        if (narrativeIframe) {
           narrativeIframe.src = narrativeGoldmineUrl;
        } else {
           logger.warn('Narrative Goldmine iframe not found. Cannot update URL.');
        }

      }, DOUBLE_CLICK_THRESHOLD);
    }
    lastClickTimeRef.current = currentTime;
  }, [graphData.nodes, onNodeDragStateChange, camera, size]);

  const handlePointerMove = useCallback((event: ThreeEvent<PointerEvent>) => {
    const drag = dragDataRef.current;
    if (!drag.isDragging || !meshRef.current) return;
    
    event.stopPropagation();
    
    // Use R3F's pointer coordinates directly
    const pointer = event.pointer;
    
    // Create a plane at the node's depth perpendicular to the camera
    const cameraDirection = new THREE.Vector3();
    camera.getWorldDirection(cameraDirection);
    
    // Create a plane at the drag start position
    const planeNormal = cameraDirection.clone().negate();
    const plane = new THREE.Plane(planeNormal, -planeNormal.dot(drag.startPosition));
    
    // Cast a ray from the camera through the mouse position
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, camera);
    
    // Find where the ray intersects the plane
    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersection);
    
    if (intersection) {
      drag.currentPosition.copy(intersection);
      
      // Update visual immediately (no React state)
      const nodeSize = settings?.visualisation?.nodes?.nodeSize || 0.01;
      const BASE_SPHERE_RADIUS = 0.5; // Ensure this matches your sphereGeometry radius
      const scale = nodeSize / BASE_SPHERE_RADIUS;
      
      const tempMatrix = new THREE.Matrix4(); // Local temp matrix
      tempMatrix.makeScale(scale, scale, scale);
      tempMatrix.setPosition(drag.currentPosition);
      meshRef.current.setMatrixAt(drag.instanceId!, tempMatrix);
      meshRef.current.instanceMatrix.needsUpdate = true;
      
      // Update the node in graphData to keep edges and labels in sync
      setGraphData(prev => ({
        ...prev,
        nodes: prev.nodes.map((node, idx) =>
          idx === drag.instanceId
            ? { ...node, position: {
                x: drag.currentPosition.x,
                y: drag.currentPosition.y,
                z: drag.currentPosition.z
              }}
            : node
        )
      }));
      
      // Prepare update for throttled send
      const now = Date.now();
      if (now - drag.lastUpdateTime > 30) { // Throttle WebSocket updates (e.g., ~30fps)
        const numericId = graphDataManager.nodeIdMap.get(drag.nodeId!);
        if (numericId !== undefined) {
          drag.pendingUpdate = {
            nodeId: numericId,
            position: {
              x: drag.currentPosition.x,
              y: drag.currentPosition.y,
              z: drag.currentPosition.z
            },
            velocity: { x: 0, y: 0, z: 0 } // Assuming velocity resets or is handled server-side
          };
          drag.lastUpdateTime = now;
        }
      }
    }
  }, [settings?.visualisation?.nodes?.nodeSize, camera, setGraphData]); // Added setGraphData to deps

  const handlePointerUp = useCallback(() => {
    const drag = dragDataRef.current;
    if (!drag.isDragging) return;
    
    // Send final position
    if (drag.nodeId && graphDataManager.webSocketService) {
      const numericId = graphDataManager.nodeIdMap.get(drag.nodeId);
      if (numericId !== undefined) {
        const finalUpdate: BinaryNodeData = {
          nodeId: numericId,
          position: {
            x: drag.currentPosition.x,
            y: drag.currentPosition.y,
            z: drag.currentPosition.z
          },
          velocity: { x: 0, y: 0, z: 0 }
        };
        graphDataManager.webSocketService.send(
          createBinaryNodeData([finalUpdate])
        );
        
        if (debugState.isEnabled()) {
          logger.debug(`Sent final position for node ${drag.nodeId}`);
        }
      }
    }
    
    dragDataRef.current.isDragging = false;
    dragDataRef.current.pendingUpdate = null; // Clear pending update
    onNodeDragStateChange(false); // <--- Signal drag end to parent
    setDragState({ nodeId: null, instanceId: null });
  }, [onNodeDragStateChange]); // Add onNodeDragStateChange to deps

  // Global pointer up listener for cases where mouse is released outside canvas
  useEffect(() => {
    const handleGlobalPointerUp = () => {
      if (dragDataRef.current.isDragging) {
        handlePointerUp();
      }
    };

    window.addEventListener('pointerup', handleGlobalPointerUp);
    return () => {
      window.removeEventListener('pointerup', handleGlobalPointerUp);
    };
  }, [handlePointerUp]);

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

    // Initial data load (now async)
    graphDataManager.getGraphData().then(initialData => {
      handleGraphDataChange(initialData)
    }).catch(error => {
      logger.error('Error getting initial graph data:', createErrorMetadata(error));
    })

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

    // Handle pending drag updates via RAF
    const drag = dragDataRef.current;
    if (drag.pendingUpdate && graphDataManager.webSocketService && graphDataManager.webSocketService.isReady()) {
      graphDataManager.webSocketService.send(
        createBinaryNodeData([drag.pendingUpdate])
      );
      drag.pendingUpdate = null; // Clear after sending
    }

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
      {/* REMOVE THE LOCAL OrbitControls INSTANCE:
      <OrbitControls ref={orbitControlsRef} ... />
      */}
      
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, graphData.nodes.length]} // Geometry, Material, Count
        frustumCulled={false}
        onPointerDown={handleNodeClick} // Corrected to use handleNodeClick
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp} // To handle release on the mesh
        onPointerMissed={() => { // To handle release outside the mesh but on canvas
          if (dragDataRef.current.isDragging) {
            handlePointerUp();
          }
        }}
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
