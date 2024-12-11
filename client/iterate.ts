import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import type { Node, Edge } from './types/core'

// Constants
const THROTTLE_INTERVAL = 16  // ~60fps max
const BINARY_VERSION = 1.0
const NODE_POSITION_SIZE = 24  // 6 floats * 4 bytes
const BINARY_HEADER_SIZE = 4   // 1 float * 4 bytes

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

// Reusable objects
const matrix = new THREE.Matrix4()
const quaternion = new THREE.Quaternion()
const position = new THREE.Vector3()
const scale = new THREE.Vector3(1, 1, 1)
const color = new THREE.Color()
const start = new THREE.Vector3()
const end = new THREE.Vector3()
const direction = new THREE.Vector3()
const center = new THREE.Vector3()
const UP = new THREE.Vector3(0, 1, 0)
const tempVector = new THREE.Vector3()

// Node state
let currentNodes: Node[] = []
let currentEdges: Edge[] = []
const NODE_SIZE = 3
const NODE_SEGMENTS = 16
const EDGE_RADIUS = 0.15
const EDGE_SEGMENTS = 8

// Colors
const NODE_COLOR = 0x4CAF50
const EDGE_COLOR = 0x90A4AE

// WebSocket state
let ws: WebSocket | null = null
let updateCount = 0
let lastUpdateTime = performance.now()
let lastMessageTime = performance.now()
const fpsHistory: number[] = []
let binaryUpdatesEnabled = false
let initialDataReceived = false
let reconnectTimeout: number | null = null
let reconnectAttempts = 0
const MAX_RECONNECT_ATTEMPTS = 5
const INITIAL_RECONNECT_DELAY = 2000
const MAX_RECONNECT_DELAY = 30000

// Message queue for throttling
interface QueuedMessage {
    data: ArrayBuffer
    timestamp: number
}
const messageQueue: QueuedMessage[] = []
let processingQueue = false

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
    scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1a)
    scene.fog = new THREE.FogExp2(0x1a1a1a, 0.002)

    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    )
    camera.position.set(0, 75, 200)
    camera.lookAt(0, 0, 0)

    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
    })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    sceneEl.appendChild(renderer.domElement)

    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.screenSpacePanning = false
    controls.minDistance = 50
    controls.maxDistance = 500

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(10, 10, 10)
    scene.add(directionalLight)

    const gridHelper = new THREE.GridHelper(1000, 100)
    if (gridHelper.material instanceof THREE.Material) {
        gridHelper.material.transparent = true
        gridHelper.material.opacity = 0.2
    }
    scene.add(gridHelper)

    initNodeMesh()
    initEdgeMesh()

    window.addEventListener('resize', onWindowResize)
    animate()
}

function initNodeMesh() {
    const geometry = new THREE.SphereGeometry(NODE_SIZE, NODE_SEGMENTS, NODE_SEGMENTS)
    const material = new THREE.MeshPhongMaterial({
        color: NODE_COLOR,
        shininess: 80,
        flatShading: false
    })

    nodeInstancedMesh = new THREE.InstancedMesh(geometry, material, 10000)
    nodeInstancedMesh.count = 0
    nodeInstancedMesh.frustumCulled = false

    scene.add(nodeInstancedMesh)
}

function initEdgeMesh() {
    const geometry = new THREE.CylinderGeometry(EDGE_RADIUS, EDGE_RADIUS, 1, EDGE_SEGMENTS)
    geometry.rotateX(Math.PI / 2)
    
    const material = new THREE.MeshBasicMaterial({
        color: EDGE_COLOR,
        transparent: true,
        opacity: 0.8,
        depthWrite: false
    })

    edgeInstancedMesh = new THREE.InstancedMesh(geometry, material, 30000)
    edgeInstancedMesh.count = 0
    edgeInstancedMesh.frustumCulled = false
    edgeInstancedMesh.renderOrder = 1

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

function createNodeInstances(nodes: Node[]) {
    currentNodes = nodes
    nodeInstancedMesh.count = nodes.length

    nodes.forEach((node, index) => {
        position.set(
            node.position?.[0] || 0,
            node.position?.[1] || 0,
            node.position?.[2] || 0
        )

        matrix.compose(position, quaternion, scale)
        nodeInstancedMesh.setMatrixAt(index, matrix)

        color.setHex(NODE_COLOR)
        nodeInstancedMesh.setColorAt(index, color)
    })

    nodeInstancedMesh.instanceMatrix.needsUpdate = true
    if (nodeInstancedMesh.instanceColor) nodeInstancedMesh.instanceColor.needsUpdate = true
}

function createEdgeInstances(nodes: Node[], edges: Edge[]) {
    currentEdges = edges
    edgeInstancedMesh.count = edges.length

    edges.forEach((edge, index) => {
        const sourceNode = nodes.find(n => n.id === edge.source)
        const targetNode = nodes.find(n => n.id === edge.target)

        if (!sourceNode || !targetNode) return

        updateEdgeInstance(index, sourceNode, targetNode)
    })

    edgeInstancedMesh.instanceMatrix.needsUpdate = true
}

function updateEdgeInstance(index: number, sourceNode: Node, targetNode: Node) {
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

    direction.subVectors(end, start)
    const length = direction.length()
    
    if (length < 0.001) return

    center.addVectors(start, end).multiplyScalar(0.5)
    position.copy(center)

    direction.normalize()
    const angle = Math.acos(THREE.MathUtils.clamp(direction.dot(UP), -1, 1))
    tempVector.crossVectors(UP, direction).normalize()
    
    if (tempVector.lengthSq() < 0.001) {
        if (direction.dot(UP) > 0) {
            quaternion.set(0, 0, 0, 1)
        } else {
            quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI)
        }
    } else {
        quaternion.setFromAxisAngle(tempVector, angle)
    }

    const adjustedLength = Math.max(0.001, length - (NODE_SIZE * 2))
    scale.set(1, adjustedLength, 1)

    matrix.compose(position, quaternion, scale)
    edgeInstancedMesh.setMatrixAt(index, matrix)
}

function updateNodePositions(floatArray: Float32Array) {
    const version = floatArray[0]
    if (version !== BINARY_VERSION) {
        log(`Warning: Received binary data version ${version}, expected ${BINARY_VERSION}`, 'warn')
    }

    const nodeCount = (floatArray.length - 1) / 6

    for (let i = 0; i < nodeCount && i < currentNodes.length; i++) {
        const baseIndex = 1 + i * 6
        position.set(
            floatArray[baseIndex],
            floatArray[baseIndex + 1],
            floatArray[baseIndex + 2]
        )

        matrix.compose(position, quaternion, scale)
        nodeInstancedMesh.setMatrixAt(i, matrix)

        if (currentNodes[i]) {
            currentNodes[i].position = [
                floatArray[baseIndex],
                floatArray[baseIndex + 1],
                floatArray[baseIndex + 2]
            ]
        }
    }

    nodeInstancedMesh.instanceMatrix.needsUpdate = true
    updateEdgePositions()
}

function updateEdgePositions() {
    currentEdges.forEach((edge, index) => {
        const sourceNode = currentNodes.find(n => n.id === edge.source)
        const targetNode = currentNodes.find(n => n.id === edge.target)

        if (!sourceNode || !targetNode) return

        updateEdgeInstance(index, sourceNode, targetNode)
    })

    edgeInstancedMesh.instanceMatrix.needsUpdate = true
}

async function processMessageQueue() {
    if (processingQueue || messageQueue.length === 0) return

    processingQueue = true
    const now = performance.now()

    while (messageQueue.length > 0) {
        const message = messageQueue[0]
        const timeSinceLastMessage = now - lastMessageTime

        if (timeSinceLastMessage < THROTTLE_INTERVAL) {
            await new Promise(resolve => setTimeout(resolve, THROTTLE_INTERVAL - timeSinceLastMessage))
        }

        const floatArray = new Float32Array(message.data)
        updateNodePositions(floatArray)
        
        updateCount++
        updatesEl.textContent = updateCount.toString()
        
        const currentTime = performance.now()
        const timeDiff = currentTime - lastUpdateTime
        const fps = 1000 / timeDiff
        
        fpsHistory.push(fps)
        if (fpsHistory.length > 10) fpsHistory.shift()
        
        const avgFps = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length
        fpsEl.textContent = avgFps.toFixed(1)
        
        lastUpdateTime = currentTime
        lastMessageTime = currentTime
        lastUpdateEl.textContent = new Date().toISOString().split('T')[1].slice(0, -1)
        
        if (updateCount % 5 === 0) {
            const nodeCount = (floatArray.length - 1) / 6
            log(`Position update #${updateCount}: ${nodeCount} nodes, ${avgFps.toFixed(1)} FPS`)
        }

        messageQueue.shift()
    }

    processingQueue = false
}

function connect() {
    if (ws?.readyState === WebSocket.CONNECTING) return

    const wsUrl = 'wss://www.visionflow.info/wss'
    
    log(`Connecting to ${wsUrl}`, 'info')
    statusEl.textContent = 'Connecting...'
    
    try {
        ws = new WebSocket(wsUrl)
        
        ws.onopen = () => {
            log('Connected to server', 'info')
            statusEl.textContent = 'Connected'
            reconnectAttempts = 0
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout)
                reconnectTimeout = null
            }
            
            if (!initialDataReceived) {
                sendMessage({ type: 'initialData' })
                log('Requested initial data', 'info')
            }
        }
        
        ws.onmessage = (event) => {
            if (event.data instanceof Blob) {
                event.data.arrayBuffer().then(buffer => {
                    if (!initialDataReceived || !binaryUpdatesEnabled) {
                        log('Ignoring binary message before initialization', 'warn')
                        return
                    }

                    const expectedSize = BINARY_HEADER_SIZE + Math.floor((buffer.byteLength - BINARY_HEADER_SIZE) / NODE_POSITION_SIZE) * NODE_POSITION_SIZE
                    if (buffer.byteLength !== expectedSize) {
                        log(`Invalid binary data length: ${buffer.byteLength} bytes (expected ${expectedSize})`, 'error')
                        return
                    }

                    messageQueue.push({
                        data: buffer,
                        timestamp: performance.now()
                    })
                    processMessageQueue()
                })
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
            binaryUpdatesEnabled = false
            initialDataReceived = false
            
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                const delay = Math.min(
                    INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttempts),
                    MAX_RECONNECT_DELAY
                )
                reconnectTimeout = window.setTimeout(connect, delay)
                reconnectAttempts++
                log(`Reconnecting in ${delay/1000} seconds (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`, 'info')
            } else {
                log('Max reconnection attempts reached', 'error')
            }
        }
    } catch (error) {
        log(`Failed to create WebSocket: ${error}`, 'error')
    }
}

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
    
    createNodeInstances(nodes)
    createEdgeInstances(nodes, edges)
    
    initialDataReceived = true
    if (!binaryUpdatesEnabled) {
        binaryUpdatesEnabled = true
        sendMessage({ type: 'enableBinaryUpdates' })
        log('Enabled binary updates', 'info')
    }
}

function cleanup() {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
    }
    
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

initThree()
connect()

window.addEventListener('unload', cleanup)
