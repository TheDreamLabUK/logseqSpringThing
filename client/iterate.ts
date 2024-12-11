import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import type { Node, Edge } from './types/core'

// Interfaces
interface GraphData {
    nodes: Node[]
    edges: Edge[]
    metadata?: any
}

interface InitialDataMessage {
    type: 'initialData'
    graphData: GraphData
    settings: any
}

interface BinaryPositionUpdateMessage {
    type: 'binaryPositionUpdate'
    is_initial_layout: boolean
}

// DOM Elements
const statusEl = document.getElementById('status') as HTMLElement
const nodeCountEl = document.getElementById('nodeCount') as HTMLElement
const edgeCountEl = document.getElementById('edgeCount') as HTMLElement
const logEl = document.getElementById('log') as HTMLElement
const fpsEl = document.getElementById('fps') as HTMLElement
const updatesEl = document.getElementById('updates') as HTMLElement
const lastUpdateEl = document.getElementById('lastUpdate') as HTMLElement
const sceneEl = document.getElementById('scene') as HTMLElement

// Three.js state
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let nodeInstancedMesh: THREE.InstancedMesh<THREE.SphereGeometry, THREE.MeshPhongMaterial>
let edgeInstancedMesh: THREE.InstancedMesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>
let animationFrameId: number

// Reusable objects for matrix operations
const matrix = new THREE.Matrix4()
const quaternion = new THREE.Quaternion()
const position = new THREE.Vector3()
const scale = new THREE.Vector3(1, 1, 1)
const color = new THREE.Color()

// Additional vectors for edge calculations
const start = new THREE.Vector3()
const end = new THREE.Vector3()
const direction = new THREE.Vector3()
const center = new THREE.Vector3()
const UP = new THREE.Vector3(0, 1, 0)

// Node state
let currentNodes: Node[] = []
let currentEdges: Edge[] = []
const NODE_SIZE = 2
const NODE_SEGMENTS = 16 // Reduced from 32 for better performance
const EDGE_RADIUS = 0.2
const EDGE_SEGMENTS = 8

// WebSocket state
let ws: WebSocket | null = null
let updateCount = 0
let lastUpdateTime = performance.now()
const fpsHistory: number[] = []

// Logging utilities
type LogType = 'log' | 'error' | 'info' | 'warn'

function log(message: string, type: LogType = 'log') {
    const div = document.createElement('div')
    div.className = type === 'warn' ? 'warning' : type
    div.textContent = `${new Date().toISOString().split('T')[1].slice(0, -1)} - ${message}`
    logEl.insertBefore(div, logEl.firstChild)
    if (logEl.children.length > 50) {
        logEl.removeChild(logEl.lastChild!)
    }
    
    switch (type) {
        case 'error':
            console.error(message)
            break
        case 'warn':
            console.warn(message)
            break
        case 'info':
            console.info(message)
            break
        default:
            console.log(message)
    }
}

// Three.js setup
function initThree() {
    // Scene
    scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1a)
    scene.fog = new THREE.FogExp2(0x1a1a1a, 0.002)

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    )
    camera.position.set(0, 75, 200)
    camera.lookAt(0, 0, 0)

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    sceneEl.appendChild(renderer.domElement)

    // Controls
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.screenSpacePanning = false
    controls.minDistance = 50
    controls.maxDistance = 500

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(10, 10, 10)
    scene.add(directionalLight)

    // Grid helper
    const gridHelper = new THREE.GridHelper(1000, 100)
    if (gridHelper.material instanceof THREE.Material) {
        gridHelper.material.transparent = true
        gridHelper.material.opacity = 0.2
    }
    scene.add(gridHelper)

    // Create instanced meshes
    initNodeMesh()
    initEdgeMesh()

    // Handle window resize
    window.addEventListener('resize', onWindowResize)

    // Start render loop
    animate()
}

function initNodeMesh() {
    // Create geometry and material
    const geometry = new THREE.SphereGeometry(NODE_SIZE, NODE_SEGMENTS, NODE_SEGMENTS)
    const material = new THREE.MeshPhongMaterial({
        color: 0x00ff00,
        shininess: 80,
        flatShading: false
    })

    // Create instanced mesh with a large enough capacity
    nodeInstancedMesh = new THREE.InstancedMesh(geometry, material, 10000)
    nodeInstancedMesh.count = 0 // Will be set when nodes are added
    nodeInstancedMesh.frustumCulled = false // Disable culling for better performance

    scene.add(nodeInstancedMesh)
}

function initEdgeMesh() {
    // Create geometry and material for edges
    const geometry = new THREE.CylinderGeometry(EDGE_RADIUS, EDGE_RADIUS, 1, EDGE_SEGMENTS)
    // Rotate cylinder to align with direction vector
    geometry.rotateX(Math.PI / 2)
    
    const material = new THREE.MeshBasicMaterial({
        color: 0x666666,
        transparent: true,
        opacity: 0.6
    })

    // Create instanced mesh with a large enough capacity
    edgeInstancedMesh = new THREE.InstancedMesh(geometry, material, 30000)
    edgeInstancedMesh.count = 0 // Will be set when edges are added
    edgeInstancedMesh.frustumCulled = false

    scene.add(edgeInstancedMesh)
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
}

function animate() {
    animationFrameId = requestAnimationFrame(animate)
    controls.update()
    renderer.render(scene, camera)
}

// Node and edge visualization
function createNodeInstances(nodes: Node[]) {
    currentNodes = nodes
    nodeInstancedMesh.count = nodes.length

    // Update all instances
    nodes.forEach((node, index) => {
        // Position
        position.set(
            node.position?.[0] || 0,
            node.position?.[1] || 0,
            node.position?.[2] || 0
        )

        // Create matrix
        matrix.compose(position, quaternion, scale)
        nodeInstancedMesh.setMatrixAt(index, matrix)

        // Set color (could be based on node properties)
        color.setHex(0x00ff00)
        nodeInstancedMesh.setColorAt(index, color)
    })

    // Mark buffers for update
    nodeInstancedMesh.instanceMatrix.needsUpdate = true
    if (nodeInstancedMesh.instanceColor) nodeInstancedMesh.instanceColor.needsUpdate = true
}

function createEdgeInstances(nodes: Node[], edges: Edge[]) {
    currentEdges = edges
    edgeInstancedMesh.count = edges.length

    // Update all edge instances
    edges.forEach((edge, index) => {
        const sourceNode = nodes.find(n => n.id === edge.source)
        const targetNode = nodes.find(n => n.id === edge.target)

        if (!sourceNode || !targetNode) return

        updateEdgeInstance(index, sourceNode, targetNode)
    })

    // Mark buffers for update
    edgeInstancedMesh.instanceMatrix.needsUpdate = true
}

function updateEdgeInstance(index: number, sourceNode: Node, targetNode: Node) {
    // Get node positions
    start.set(
        sourceNode.position?.[0] || 0,
        sourceNode.position?.[1] || 0,
        sourceNode.position?.[2] || 0
    )
    end.set(
        targetNode.position?.[0] || 0,
        targetNode.position?.[1] || 0,
        targetNode.position?.[2] || 0
    )

    // Calculate edge transform
    direction.subVectors(end, start)
    const length = direction.length()
    
    // Skip if nodes are in same position
    if (length === 0) return

    // Position at midpoint
    center.addVectors(start, end).multiplyScalar(0.5)
    position.copy(center)

    // Scale to length
    scale.set(1, length, 1)

    // Rotate to align with direction
    direction.normalize()
    quaternion.setFromUnitVectors(UP, direction)

    // Update matrix
    matrix.compose(position, quaternion, scale)
    edgeInstancedMesh.setMatrixAt(index, matrix)
}

function updateNodePositions(floatArray: Float32Array) {
    const nodeCount = (floatArray.length - 1) / 6 // Subtract 1 for version header

    // Update positions for each node
    for (let i = 0; i < nodeCount; i++) {
        const baseIndex = 1 + i * 6 // Skip version header
        position.set(
            floatArray[baseIndex],
            floatArray[baseIndex + 1],
            floatArray[baseIndex + 2]
        )

        // Update node matrix
        matrix.compose(position, quaternion, scale)
        nodeInstancedMesh.setMatrixAt(i, matrix)

        // Update node data
        if (currentNodes[i]) {
            currentNodes[i].position = [
                floatArray[baseIndex],
                floatArray[baseIndex + 1],
                floatArray[baseIndex + 2]
            ]
        }
    }

    // Mark node matrix for update
    nodeInstancedMesh.instanceMatrix.needsUpdate = true

    // Update edge positions
    updateEdgePositions()
}

function updateEdgePositions() {
    currentEdges.forEach((edge, index) => {
        const sourceNode = currentNodes.find(n => n.id === edge.source)
        const targetNode = currentNodes.find(n => n.id === edge.target)

        if (!sourceNode || !targetNode) return

        updateEdgeInstance(index, sourceNode, targetNode)
    })

    // Mark edge matrix for update
    edgeInstancedMesh.instanceMatrix.needsUpdate = true
}

// WebSocket connection
function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/wss`
    
    log(`Connecting to ${wsUrl}`, 'info')
    statusEl.textContent = 'Connecting...'
    
    try {
        ws = new WebSocket(wsUrl)
        
        ws.onopen = () => {
            log('Connected to server', 'info')
            statusEl.textContent = 'Connected'
            
            // Request initial data
            sendMessage({ type: 'initialData' })
            log('Requested initial data', 'info')
        }
        
        ws.onmessage = (event) => {
            if (event.data instanceof Blob) {
                handleBinaryMessage(event.data)
            } else {
                handleJsonMessage(event.data)
            }
        }
        
        ws.onerror = (error) => {
            log(`WebSocket error: ${error}`, 'error')
            statusEl.textContent = 'Error'
        }
        
        ws.onclose = () => {
            log('Disconnected from server', 'warn')
            statusEl.textContent = 'Disconnected'
            ws = null
            
            // Attempt to reconnect after 2 seconds
            setTimeout(connect, 2000)
        }
    } catch (error) {
        log(`Failed to create WebSocket: ${error}`, 'error')
    }
}

// Helper function to safely send messages
function sendMessage(data: unknown) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('Cannot send message: WebSocket is not connected', 'warn')
        return false
    }
    
    try {
        ws.send(JSON.stringify(data))
        return true
    } catch (error) {
        log(`Failed to send message: ${error}`, 'error')
        return false
    }
}

// Message handlers
function handleJsonMessage(data: string) {
    try {
        const message = JSON.parse(data)
        log(`Received message type: ${message.type}`, 'info')
        
        switch (message.type) {
            case 'initialData':
                handleInitialData(message as InitialDataMessage)
                break
            case 'binaryPositionUpdate':
                log('Binary position update incoming')
                break
            default:
                log(`Unknown message type: ${message.type}`, 'warn')
        }
    } catch (error) {
        log(`Failed to parse message: ${error}`, 'error')
    }
}

function handleInitialData(message: InitialDataMessage) {
    const { nodes, edges } = message.graphData
    nodeCountEl.textContent = nodes.length.toString()
    edgeCountEl.textContent = edges.length.toString()
    
    log(`Received initial data: ${nodes.length} nodes, ${edges.length} edges`)
    
    // Create instances
    createNodeInstances(nodes)
    createEdgeInstances(nodes, edges)
    
    // Enable binary updates
    sendMessage({ type: 'enableBinaryUpdates' })
    log('Enabled binary updates', 'info')
}

function handleBinaryMessage(data: Blob) {
    const reader = new FileReader()
    reader.onload = () => {
        const buffer = reader.result as ArrayBuffer
        const floatArray = new Float32Array(buffer)
        const nodeCount = (floatArray.length - 1) / 6 // Subtract 1 for version header
        
        // Update node positions
        updateNodePositions(floatArray)
        
        updateCount++
        updatesEl.textContent = updateCount.toString()
        
        const now = performance.now()
        const timeDiff = now - lastUpdateTime
        const fps = 1000 / timeDiff
        
        fpsHistory.push(fps)
        if (fpsHistory.length > 10) fpsHistory.shift()
        
        const avgFps = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length
        fpsEl.textContent = avgFps.toFixed(1)
        
        lastUpdateTime = now
        lastUpdateEl.textContent = new Date().toISOString().split('T')[1].slice(0, -1)
        
        if (updateCount % 5 === 0) {
            log(`Position update #${updateCount}: ${nodeCount} nodes, ${avgFps.toFixed(1)} FPS`)
        }
    }
    reader.readAsArrayBuffer(data)
}

// Cleanup function
function cleanup() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
    }
    
    if (nodeInstancedMesh) {
        nodeInstancedMesh.geometry.dispose()
        if (nodeInstancedMesh.material instanceof THREE.Material) {
            nodeInstancedMesh.material.dispose()
        }
        scene.remove(nodeInstancedMesh)
    }

    if (edgeInstancedMesh) {
        edgeInstancedMesh.geometry.dispose()
        if (edgeInstancedMesh.material instanceof THREE.Material) {
            edgeInstancedMesh.material.dispose()
        }
        scene.remove(edgeInstancedMesh)
    }
    
    if (renderer) {
        renderer.dispose()
    }
    
    if (controls) {
        controls.dispose()
    }
    
    window.removeEventListener('resize', onWindowResize)
}

// Initialize everything
initThree()
connect()

// Cleanup on window unload
window.addEventListener('unload', cleanup)
