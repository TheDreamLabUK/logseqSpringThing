import { ref, watch, onBeforeUnmount, markRaw } from 'vue'
import { useVisualizationStore } from '../stores/visualization'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import { useSettingsStore } from '../stores/settings'
import { useForceGraph } from './useForceGraph'
import { useThreeScene } from './useThreeScene'
import { Group, Vector3 } from 'three'
import type { Camera } from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import type { WebXRVisualizationState } from '../types/visualization'
import type { Node, Edge } from '../types/core'

// Performance tuning constants
const MIN_UPDATE_INTERVAL = 16 // ~60fps
const BATCH_UPDATE_SIZE = 1000 // Number of nodes to update per frame
const FRAME_BUDGET_MS = 16 // Target 60fps
const PERFORMANCE_SAMPLE_SIZE = 60 // Number of frames to average for performance metrics
const ADAPTIVE_BATCH_THRESHOLD = 0.9 // Threshold for batch size adjustment
const MIN_BATCH_SIZE = 100 // Minimum batch size
const MAX_BATCH_SIZE = 5000 // Maximum batch size
const PERFORMANCE_LOG_THRESHOLD = 5 // Log if performance changes by this many ms

interface PerformanceMetrics {
  frameTimeHistory: number[]
  updateTimeHistory: number[]
  averageFrameTime: number
  averageUpdateTime: number
  lastFrameTime: number
  lastUpdateTime: number
  frameTimeDeviation: number
  batchSize: number
  droppedFrames: number
  lastPerformanceLog: number
}

interface GraphSystemState {
  initialized: boolean
  lastUpdateTime: number
  frameCount: number
  pendingUpdates: boolean
  updateInProgress: boolean
  performanceMetrics: PerformanceMetrics
}

// Helper function to safely handle OrbitControls
const createNonReactiveControls = (controls: OrbitControls | null): OrbitControls | null => {
  return controls ? markRaw(controls) : null;
};

export function useGraphSystem() {
  // Stores
  const visualizationStore = useVisualizationStore()
  const binaryUpdateStore = useBinaryUpdateStore()
  const settingsStore = useSettingsStore()
  const error = ref<Error | null>(null)

  // Initialize Three.js scene
  const { resources, initScene, dispose: disposeScene } = useThreeScene()

  // Graph groups
  const graphGroup = markRaw(new Group())
  const nodesGroup = markRaw(new Group())
  const edgesGroup = markRaw(new Group())
  graphGroup.add(nodesGroup)
  graphGroup.add(edgesGroup)

  // State
  const visualizationState = ref<WebXRVisualizationState>({
    initialized: false,
    pendingInitialization: false,
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    xrSessionManager: null,
    canvas: null
  })

  const systemState = ref<GraphSystemState>({
    initialized: false,
    lastUpdateTime: 0,
    frameCount: 0,
    pendingUpdates: false,
    updateInProgress: false,
    performanceMetrics: {
      frameTimeHistory: [],
      updateTimeHistory: [],
      averageFrameTime: 0,
      averageUpdateTime: 0,
      lastFrameTime: 0,
      lastUpdateTime: 0,
      frameTimeDeviation: 0,
      batchSize: BATCH_UPDATE_SIZE,
      droppedFrames: 0,
      lastPerformanceLog: 0
    }
  })

  // Node interaction state
  const hoveredNode = ref<string | null>(null)

  // Initialize force graph
  let forceGraph: ReturnType<typeof useForceGraph> | null = null
  let rafHandle: number | null = null

  // Node helpers
  const getNodePosition = (node: Node | string): Vector3 => {
    const id = typeof node === 'object' ? node.id : node
    const nodeData = visualizationStore.nodes.find(n => n.id === id)
    if (!nodeData?.position) return markRaw(new Vector3())
    return markRaw(new Vector3(nodeData.position[0], nodeData.position[1], nodeData.position[2]))
  }

  const getNodeScale = (node: Node): number => {
    const settings = settingsStore.getVisualizationSettings
    const baseSize = node.size || 1
    const minSize = settings.min_node_size
    const maxSize = settings.max_node_size
    return minSize + (baseSize * (maxSize - minSize))
  }

  const getNodeColor = (node: Node): string => {
    return node.id === hoveredNode.value
      ? settingsStore.getVisualizationSettings.node_color_core
      : (node.color || settingsStore.getVisualizationSettings.node_color)
  }

  // Edge helpers
  const getEdgePoints = (source: Node, target: Node): [Vector3, Vector3] => {
    return [
      getNodePosition(source),
      getNodePosition(target)
    ]
  }

  const getEdgeColor = (edge: Edge): string => {
    return edge.color || settingsStore.getVisualizationSettings.edge_color
  }

  const getEdgeWidth = (edge: Edge): number => {
    const settings = settingsStore.getVisualizationSettings
    const baseWidth = edge.weight || 1
    const minWidth = settings.edge_min_width
    const maxWidth = settings.edge_max_width
    return minWidth + (baseWidth * (maxWidth - minWidth))
  }

  // Performance metrics update
  const updatePerformanceMetrics = (frameTime: number, updateTime: number): void => {
    const metrics = systemState.value.performanceMetrics

    // Update histories
    metrics.frameTimeHistory.push(frameTime)
    metrics.updateTimeHistory.push(updateTime)

    // Keep histories at fixed size
    if (metrics.frameTimeHistory.length > PERFORMANCE_SAMPLE_SIZE) {
      metrics.frameTimeHistory.shift()
      metrics.updateTimeHistory.shift()
    }

    // Calculate averages
    metrics.averageFrameTime = metrics.frameTimeHistory.reduce((a: number, b: number) => a + b, 0) / metrics.frameTimeHistory.length
    metrics.averageUpdateTime = metrics.updateTimeHistory.reduce((a: number, b: number) => a + b, 0) / metrics.updateTimeHistory.length

    // Calculate frame time deviation
    const variance = metrics.frameTimeHistory.reduce((acc: number, time: number) => {
      const diff = time - metrics.averageFrameTime
      return acc + (diff * diff)
    }, 0) / metrics.frameTimeHistory.length
    metrics.frameTimeDeviation = Math.sqrt(variance)

    // Track dropped frames
    if (frameTime > FRAME_BUDGET_MS * 1.5) {
      metrics.droppedFrames++
    }

    // Adapt batch size based on performance
    if (metrics.averageFrameTime > FRAME_BUDGET_MS * ADAPTIVE_BATCH_THRESHOLD) {
      // Reduce batch size if we're exceeding frame budget
      metrics.batchSize = Math.max(MIN_BATCH_SIZE, metrics.batchSize * 0.9)
    } else if (metrics.averageFrameTime < FRAME_BUDGET_MS * 0.7 && metrics.frameTimeDeviation < 2) {
      // Increase batch size if we have stable performance and room in the frame budget
      metrics.batchSize = Math.min(MAX_BATCH_SIZE, metrics.batchSize * 1.1)
    }

    // Update last times
    metrics.lastFrameTime = frameTime
    metrics.lastUpdateTime = updateTime

    // Log performance metrics if they've changed significantly
    const now = performance.now()
    if (Math.abs(metrics.averageFrameTime - metrics.lastFrameTime) > PERFORMANCE_LOG_THRESHOLD &&
        now - metrics.lastPerformanceLog > 1000) {
      console.debug('Performance metrics:', {
        averageFrameTime: metrics.averageFrameTime.toFixed(2),
        frameTimeDeviation: metrics.frameTimeDeviation.toFixed(2),
        batchSize: Math.floor(metrics.batchSize),
        droppedFrames: metrics.droppedFrames,
        timestamp: new Date().toISOString()
      })
      metrics.lastPerformanceLog = now
    }
  }

  // Update control
  const shouldSkipUpdate = (): boolean => {
    const now = performance.now()
    const timeSinceLastUpdate = now - systemState.value.lastUpdateTime
    
    // Skip if updating too frequently
    if (timeSinceLastUpdate < MIN_UPDATE_INTERVAL) {
      return true
    }

    // Skip if previous update is still in progress
    if (systemState.value.updateInProgress) {
      systemState.value.pendingUpdates = true
      return true
    }

    return false
  }

  // Node interaction handlers
  const handleNodeClick = (node: Node): void => {
    const position = getNodePosition(node)
    console.debug('Node clicked:', { 
      id: node.id, 
      position: position.toArray()
    })
  }

  const handleNodeHover = (node: Node | null): void => {
    hoveredNode.value = node?.id || null
    if (visualizationState.value.scene) {
      visualizationState.value.scene.userData.needsRender = true
    }
  }

  // Resource cleanup
  const dispose = (): void => {
    // Cancel any pending animation frame
    if (rafHandle !== null) {
      cancelAnimationFrame(rafHandle)
      rafHandle = null
    }

    // Clean up force graph
    if (forceGraph) {
      forceGraph.dispose()
      forceGraph = null
    }

    // Clean up scene
    if (resources.value) {
      disposeScene()
    }

    // Reset state
    visualizationState.value = {
      initialized: false,
      pendingInitialization: false,
      scene: null,
      camera: null,
      renderer: null,
      controls: null,
      xrSessionManager: null,
      canvas: null
    }

    systemState.value = {
      initialized: false,
      lastUpdateTime: 0,
      frameCount: 0,
      pendingUpdates: false,
      updateInProgress: false,
      performanceMetrics: {
        frameTimeHistory: [],
        updateTimeHistory: [],
        averageFrameTime: 0,
        averageUpdateTime: 0,
        lastFrameTime: 0,
        lastUpdateTime: 0,
        frameTimeDeviation: 0,
        batchSize: BATCH_UPDATE_SIZE,
        droppedFrames: 0,
        lastPerformanceLog: 0
      }
    }
  }

  // System initialization
  const initialize = async (): Promise<void> => {
    if (systemState.value.initialized) return

    try {
      error.value = null // Reset error state
      
      // Initialize Three.js scene
      const sceneResources = await initScene()
      if (!sceneResources) {
        throw new Error('Failed to initialize Three.js scene')
      }

      // Handle controls separately to maintain type safety
      const controls = createNonReactiveControls(sceneResources.controls as OrbitControls | null);
      
      // Update visualization state with raw THREE.js objects
      visualizationState.value = {
        initialized: true,
        pendingInitialization: false,
        scene: markRaw(sceneResources.scene),
        camera: markRaw(sceneResources.camera),
        renderer: markRaw(sceneResources.renderer),
        controls,
        xrSessionManager: null,
        canvas: markRaw(sceneResources.renderer.domElement)
      }
      
      // Initialize force graph
      if (!visualizationState.value.scene) {
        throw new Error('Scene not initialized')
      }

      forceGraph = useForceGraph(visualizationState.value.scene)
      if (!forceGraph) {
        throw new Error('Failed to initialize force graph')
      }

      systemState.value.initialized = true
      console.debug('Graph system initialized:', {
        state: systemState.value,
        visualization: visualizationState.value,
        forceGraph: !!forceGraph
      })

    } catch (err: unknown) {
      error.value = err instanceof Error ? err : new Error('Unknown error during initialization')
      console.error('Failed to initialize graph system:', error.value)
      visualizationState.value.initialized = false
      systemState.value.initialized = false
      throw error.value // Re-throw for error boundary
    }
  }

  // Graph data management
  const updateGraphData = (graphData: { nodes: Node[]; edges: Edge[] }): void => {
    if (systemState.value.updateInProgress) return
    if (!graphData.nodes || !Array.isArray(graphData.nodes)) {
      console.error('Invalid nodes data:', graphData.nodes)
      return
    }
    if (!graphData.edges || !Array.isArray(graphData.edges)) {
      console.error('Invalid edges data:', graphData.edges)
      return
    }
    
    systemState.value.updateInProgress = true
    try {
      error.value = null // Reset error state

      // Validate nodes
      const validNodes = graphData.nodes.filter(node => {
        if (!node.id) {
          console.warn('Node missing ID:', node)
          return false
        }
        return true
      })

      // Validate edges
      const validEdges = graphData.edges.filter(edge => {
        if (!edge.source || !edge.target) {
          console.warn('Edge missing source or target:', edge)
          return false
        }
        return true
      })

      // Update store data
      visualizationStore.nodes = validNodes
      visualizationStore.edges = validEdges

      // Update force graph
      if (forceGraph) {
        forceGraph.updateGraph(validNodes, validEdges)
      }

      // Mark for re-render
      if (visualizationState.value.scene) {
        visualizationState.value.scene.userData.needsRender = true
      }

      console.debug('Graph data updated:', {
        nodes: validNodes.length,
        edges: validEdges.length,
        invalidNodes: graphData.nodes.length - validNodes.length,
        invalidEdges: graphData.edges.length - validEdges.length
      })

    } catch (err: unknown) {
      error.value = err instanceof Error ? err : new Error('Failed to update graph data')
      console.error('Error updating graph data:', error.value)
    } finally {
      systemState.value.updateInProgress = false
    }
  }

  const updateNodePosition = (
    id: string,
    position: Vector3,
    velocity: Vector3
  ): void => {
    if (systemState.value.updateInProgress) return
    
    const index = visualizationStore.nodes.findIndex(node => node.id === id)
    if (index === -1) return

    systemState.value.updateInProgress = true
    try {
      // Update binary store
      binaryUpdateStore.updateNodePosition(
        index,
        position.x,
        position.y,
        position.z,
        velocity.x,
        velocity.y,
        velocity.z
      )

      // Mark for re-render
      if (visualizationState.value.scene) {
        visualizationState.value.scene.userData.needsRender = true
      }
    } finally {
      systemState.value.updateInProgress = false
    }
  }

  // Visualization update
  const updateVisualization = async (): Promise<void> => {
    if (!visualizationState.value.scene || !forceGraph) return
    if (shouldSkipUpdate()) return

    const startTime = performance.now()
    systemState.value.updateInProgress = true

    try {
      const camera = visualizationState.value.camera as Camera
      if (!camera) return

      // Get changed nodes from binary update store
      const changedNodes = binaryUpdateStore.getChangedNodes
      
      // Update graph data if nodes have changed
      if (changedNodes.size > 0) {
        const metrics = systemState.value.performanceMetrics
        const batchSize = Math.floor(metrics.batchSize)
        
        // Update nodes in batches to maintain frame rate
        const nodesToUpdate = Array.from(changedNodes)
        for (let i = 0; i < nodesToUpdate.length; i += batchSize) {
          const batchStartTime = performance.now()
          
          // Process this batch
          const batch = nodesToUpdate.slice(i, i + batchSize)
          
          // Update visualization for this batch
          forceGraph.updateNodes(camera)
          forceGraph.updateLinks(camera)

          // Update performance metrics
          const batchEndTime = performance.now()
          const batchTime = batchEndTime - batchStartTime
          updatePerformanceMetrics(batchTime, batchTime)

          // Check if we've exceeded frame budget
          if (batchEndTime - startTime > FRAME_BUDGET_MS) {
            // Schedule remaining updates for next frame
            systemState.value.pendingUpdates = true
            break
          }

          // Small delay to allow other operations
          await new Promise(resolve => setTimeout(resolve, 0))
        }

        // Mark scene for re-render
        visualizationState.value.scene.userData.needsRender = true
      }

      // Update timing state
      systemState.value.lastUpdateTime = performance.now()
      systemState.value.frameCount++

    } finally {
      systemState.value.updateInProgress = false
    }

    // Schedule next update if there are pending changes
    if (systemState.value.pendingUpdates) {
      systemState.value.pendingUpdates = false
      rafHandle = requestAnimationFrame(updateVisualization)
    }
  }

  // Initialize scene groups with error handling
  watch(visualizationState, (state) => {
    try {
      if (state.scene && !state.scene.userData.graphGroup) {
        state.scene.add(graphGroup)
        state.scene.userData.graphGroup = graphGroup
        state.scene.userData.nodesGroup = nodesGroup
        state.scene.userData.edgesGroup = edgesGroup
        console.debug('Scene groups initialized')
      }
    } catch (err: unknown) {
      error.value = err instanceof Error ? err : new Error('Failed to initialize scene groups')
      console.error('Error initializing scene groups:', error.value)
    }
  }, { deep: true })

  // Watch for binary updates
  watch(() => binaryUpdateStore.getAllPositions, () => {
    if (!systemState.value.initialized) return
    
    const nodeCount = visualizationStore.nodes.length
    if (nodeCount === 0) return

    // Schedule visualization update
    rafHandle = requestAnimationFrame(updateVisualization)
  }, { deep: true })

  // Watch for graph data changes
  watch(() => visualizationStore.nodes, (newNodes) => {
    if (!systemState.value.initialized || !forceGraph) return
    if (newNodes.length === 0) return

    // Update entire graph when data changes
    forceGraph.updateGraph(
      visualizationStore.nodes,
      visualizationStore.edges
    )

    // Force immediate update
    systemState.value.lastUpdateTime = 0
    rafHandle = requestAnimationFrame(updateVisualization)
  }, { deep: true })

  // Watch for settings changes
  watch(() => settingsStore.getVisualizationSettings, async () => {
    if (!systemState.value.initialized) return

    // Reinitialize with new settings
    dispose()
    await initialize()

    // Force update with new settings
    if (forceGraph) {
      forceGraph.updateGraph(
        visualizationStore.nodes,
        visualizationStore.edges
      )
      systemState.value.lastUpdateTime = 0
      rafHandle = requestAnimationFrame(updateVisualization)
    }
  }, { deep: true })

  // Initialize on creation
  void initialize()

  // Clean up on unmount
  onBeforeUnmount(() => {
    dispose()
  })

  return {
    visualizationState,
    systemState,
    error,
    initialize,
    updateVisualization,
    dispose,
    // Graph structure
    graphGroup,
    nodesGroup,
    edgesGroup,
    // Node state
    hoveredNode,
    // Node helpers
    getNodePosition,
    getNodeScale,
    getNodeColor,
    // Edge helpers
    getEdgePoints,
    getEdgeColor,
    getEdgeWidth,
    // Interaction handlers
    handleNodeClick,
    handleNodeHover,
    // Data management
    updateGraphData,
    updateNodePosition
  }
}
