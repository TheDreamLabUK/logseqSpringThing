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

interface NodeMesh extends THREE.Mesh {
    material: THREE.Material
    geometry: THREE.BufferGeometry
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
let nodeObjects: NodeMesh[] = []
let animationFrameId: number

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

    // Handle window resize
    window.addEventListener('resize', onWindowResize)

    // Start render loop
    animate()
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

// Node visualization
function createNodeObjects(nodes: Node[]) {
    // Clear existing nodes
    nodeObjects.forEach(node => {
        scene.remove(node)
        node.geometry.dispose()
        node.material.dispose()
    })
    nodeObjects = []

    // Create sphere geometry and material
    const geometry = new THREE.SphereGeometry(2, 32, 32)
    const material = new THREE.MeshPhongMaterial({ color: 0x00ff00 })

    // Create a mesh for each node
    nodes.forEach(node => {
        const mesh = new THREE.Mesh(geometry, material) as NodeMesh
        mesh.position.set(
            node.position?.[0] || 0,
            node.position?.[1] || 0,
            node.position?.[2] || 0
        )
        scene.add(mesh)
        nodeObjects.push(mesh)
    })
}

function updateNodePositions(floatArray: Float32Array) {
    // Skip version header (first float)
    for (let i = 0; i < nodeObjects.length; i++) {
        const baseIndex = 1 + i * 6 // Skip version header, 6 floats per node (x,y,z, vx,vy,vz)
        nodeObjects[i].position.set(
            floatArray[baseIndex],
            floatArray[baseIndex + 1],
            floatArray[baseIndex + 2]
        )
    }
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
    
    // Create node objects in the scene
    createNodeObjects(nodes)
    
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
        
        // Update node positions in the scene
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
    
    nodeObjects.forEach(node => {
        scene.remove(node)
        node.geometry.dispose()
        node.material.dispose()
    })
    nodeObjects = []
    
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
