import { useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';

// Components
 import GraphManager from './GraphManager';
import XRController from '../../xr/components/XRController';
import XRVisualizationConnector from '../../xr/components/XRVisualizationConnector';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/logger';
import { debugState } from '../../../utils/debugState';

const logger = createLogger('GraphCanvas');

// Scene setup with lighting and background
const SceneSetup = () => {
    const { scene } = useThree();
    const settings = useSettingsStore(state => state.settings?.visualization);

    // Render lights using JSX
    return (
        <>
            <color attach="background" args={[0, 0, 0.8]} /> {/* Medium blue background */}
            <ambientLight intensity={0.6} />
            <directionalLight 
                intensity={0.8} 
                position={[1, 1, 1]} 
            />
        </>
    );
};

// Main GraphCanvas component
const GraphCanvas = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { settings } = useSettingsStore();
    const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false; // Use performance debug flag
    const xrEnabled = settings?.xr?.enabled !== false;
    const antialias = settings?.visualization?.rendering?.enableAntialiasing !== false; // Correct property name

    return (
        <div 
            className="absolute inset-0 overflow-hidden"
            style={{
                width: '100%',
                height: '100%',
                minHeight: '0',
                display: 'flex',
                flexDirection: 'column'
            }}
        >
            <Canvas
                ref={canvasRef}
                style={{
                    width: '100%',
                    height: '100%',
                    display: 'block',
                    minHeight: '0',
                    flex: '1 1 auto'
                }}
                gl={{
                    antialias,
                    alpha: true,
                    powerPreference: 'high-performance',
                    failIfMajorPerformanceCaveat: false
                }}
                camera={{
                    fov: 75,
                    near: 0.1,
                    far: 2000, // Remove settings access, camera settings likely managed elsewhere
                    position: [0, 10, 50]
                }}
                className="r3f-canvas"
                onCreated={({ gl }) => {
                    if (debugState.isEnabled()) {
                        logger.debug('Canvas created with dimensions:', {
                            width: gl.domElement.width,
                            height: gl.domElement.height,
                            containerWidth: gl.domElement.parentElement?.clientWidth,
                            containerHeight: gl.domElement.parentElement?.clientHeight
                        });
                    }
                }}
            >
                <SceneSetup />
                <GraphManager />
                {xrEnabled && <XRController />}
                {xrEnabled && <XRVisualizationConnector />}
                <OrbitControls
                    enableDamping={true}
                    dampingFactor={0.1}
                    screenSpacePanning={true}
                    minDistance={1}
                    maxDistance={2000}
                    enableRotate={true}
                    enableZoom={true}
                    enablePan={true}
                    rotateSpeed={1.0}
                    zoomSpeed={1.2}
                    panSpeed={0.8}
                />
                {showStats && <Stats />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;
