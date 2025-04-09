import React, { useEffect, useState, Suspense, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// Visualization components
import GraphManager from '../components/graph/GraphManager';
import { graphDataManager } from '../lib/managers/graph-data-manager';

// Layout components
import ViewportContainer from '../components/layout/ViewportContainer';
import MainLayout from '../components/layout/MainLayout';
import ViewportControls from '../components/viewport/ViewportControls';

// Panels and UI components
import Panel from '../components/panel/Panel';
import DockingZone from '../components/panel/DockingZone';
import SystemPanel from '../components/settings/panels/SystemPanel';
import { Toaster } from '../components/ui/toaster';

// Contexts and providers
import { ThemeProvider } from '../components/ui/theme-provider';
import { TooltipProvider } from '../components/ui/tooltip';
import { PanelProvider } from '../components/panel/PanelContext';
import { WindowSizeProvider } from '../lib/contexts/WindowSizeContext';
import { ApplicationModeProvider } from '../components/context/ApplicationModeContext';
import SafeXRProvider from '../lib/xr/SafeXRProvider';

// Utilities
import { useSettingsStore } from '../lib/stores/settings-store';
import { createLogger } from '../lib/utils/logger';
import AppInitializer from '../components/AppInitializer';

const logger = createLogger('ModernAppPage');

// Main application component
const ModernAppPage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50);
  const orbitControlsRef = useRef<any>(null);
  const { initialized } = useSettingsStore(state => ({
    initialized: state.initialized
  }));

  // Initialize graph data and camera position when component mounts
  useEffect(() => {
    const initializeGraph = async () => {
      try {
        await graphDataManager.fetchInitialData();
        
        // Calculate graph center and size
        const data = graphDataManager.getGraphData();
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        data.nodes.forEach(node => {
          if (node.position) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
            minZ = Math.min(minZ, node.position.z);
            maxZ = Math.max(maxZ, node.position.z);
          }
        });
        
        const centerX = (maxX + minX) / 2;
        const centerY = (maxY + minY) / 2;
        const centerZ = (maxZ + minZ) / 2;
        
        const width = maxX - minX;
        const height = maxY - minY;
        const depth = maxZ - minZ;
        
        const maxDimension = Math.max(width, height, depth);
        
        setGraphCenter([centerX, centerY, centerZ]);
        setGraphSize(maxDimension);
        
        if (initialized) {
          setIsLoading(false);
        }
      } catch (err) {
        logger.error('Failed to fetch graph data', err);
        if (initialized) {
          setIsLoading(false);
        }
      }
    };

    initializeGraph();
  }, [initialized]);

  // Initialize settings store
  useEffect(() => {
    if (initialized) {
      setIsLoading(false);
    }
  }, [initialized]);

  const handleInitialized = () => {
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('Application initialized');
    }
    setIsLoading(false);
  };
  
  // Camera controller component
  const CameraController = ({ center, size }: { center: [number, number, number], size: number }) => {
    const distanceFactor = size * 1.5;
    return null; // Just for prop types, OrbitControls does the actual work
  };
  
  // Handler functions
  const handleResetCamera = () => {
    logger.debug('Reset camera');
    if (orbitControlsRef.current) {
      orbitControlsRef.current.reset();
      orbitControlsRef.current.target.set(graphCenter[0], graphCenter[1], graphCenter[2]);
      orbitControlsRef.current.update();
    }
  };
  
  const handleZoomIn = () => {
    logger.debug('Zoom in');
    if (orbitControlsRef.current) {
      const currentDistance = orbitControlsRef.current.getDistance();
      const newDistance = Math.max(currentDistance * 0.8, 1);
      orbitControlsRef.current.dollyTo(newDistance, true);
    }
  };
  
  const handleZoomOut = () => {
    logger.debug('Zoom out');
    if (orbitControlsRef.current) {
      const currentDistance = orbitControlsRef.current.getDistance();
      const newDistance = Math.min(currentDistance * 1.2, 2000);
      orbitControlsRef.current.dollyTo(newDistance, true);
    }
  };
  
  const handleToggleFullscreen = () => {
    logger.debug('Toggle fullscreen');
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        logger.error('Error attempting to enable fullscreen:', err);
      });
    } else {
      document.exitFullscreen();
    }
  };
  
  const handleRotateView = () => {
    logger.debug('Rotate view');
    if (orbitControlsRef.current) {
      const currentRotation = orbitControlsRef.current.getAzimuthalAngle();
      orbitControlsRef.current.setAzimuthalAngle(currentRotation + Math.PI / 2);
    }
  };
  
  const handleToggleRightPanel = () => {
    setShowRightPanel(!showRightPanel);
  };

  // Background color 
  const backgroundColor = '#000000'; // Black background

  return (
    <ThemeProvider defaultTheme="dark">
      <WindowSizeProvider>
        <ApplicationModeProvider>
          <PanelProvider>
            <TooltipProvider>
              <SafeXRProvider>
                <div className="app-container w-full h-full flex flex-col overflow-hidden" style={{ height: '100vh' }}>
                  <MainLayout
                    viewportContent={
                      <ViewportContainer>
                        <div className="relative w-full h-full overflow-hidden" style={{ height: '100%', minHeight: '100%' }}>
                          {/* Three.js Canvas */}
                          <div className="absolute inset-0" style={{
                            display: isLoading ? 'none' : 'block',
                            height: '100%'
                          }}>
                            <div className="relative w-full h-full" style={{ height: '100%' }}>
                              <Canvas
                                className="three-canvas"
                                style={{
                                  display: 'block',
                                  position: 'absolute',
                                width: '100%',
                                height: '100%'
                              }}
                              camera={{
                                position: [0, 0, 50],
                                fov: 50,
                                near: 0.1,
                                far: 2000
                              }}
                              gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
                              dpr={[1, 2]}
                              shadows
                            >
                              <color attach="background" args={[backgroundColor]} />
                              <CameraController center={graphCenter} size={graphSize} />
                              <ambientLight intensity={0.6} />
                              <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
                              <pointLight position={[-10, -10, -5]} intensity={0.5} />
                              <OrbitControls
                                ref={orbitControlsRef}
                                makeDefault
                                enableDamping
                                dampingFactor={0.05}
                                minDistance={1}
                                maxDistance={2000}
                                target={[graphCenter[0], graphCenter[1], graphCenter[2]]}
                              />
                              <Suspense fallback={null}>
                                <GraphManager />
                              </Suspense>
                              {/* Debug axes helper */}
                              <axesHelper args={[5]} />
                              <Stats />
                              </Canvas>
                            </div>
                          </div>
                          
                          {/* Loading Overlay */}
                          {isLoading && (
                            <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
                              <div className="flex flex-col items-center space-y-4">
                                <div className="text-2xl">Loading Graph Visualization</div>
                                <div className="h-2 w-48 overflow-hidden rounded-full bg-gray-700">
                                  <div className="animate-pulse h-full bg-primary"></div>
                                </div>
                              </div>
                            </div>
                          )}
                          
                          {/* ViewportControls - positioned using fixed in the component */}
                          {!isLoading && (
                            <ViewportControls
                              onReset={handleResetCamera}
                              onZoomIn={handleZoomIn}
                              onZoomOut={handleZoomOut}
                              onToggleFullscreen={handleToggleFullscreen}
                              onRotate={handleRotateView}
                              onToggleRightPanel={handleToggleRightPanel}
                            />
                          )}
                        </div>
                      </ViewportContainer>
                    }
                    rightDockContent={
                      !isLoading && showRightPanel && (
                        <DockingZone
                          position="right"
                          className="active"
                          defaultSize={350}
                          expandable={true}
                          autoSize={false}
                        >
                          <Panel id="system-right">
                            <SystemPanel panelId="system-right" />
                          </Panel>
                        </DockingZone>
                      )
                    }
                  />
                  
                  {/* Control instructions overlay - similar to SimpleGraphPage */}
                  <div style={{ 
                    position: 'absolute', 
                    bottom: '10px', 
                    left: '10px', 
                    color: 'white', 
                    zIndex: 1000,
                    background: 'rgba(0,0,0,0.5)',
                    padding: '10px',
                    borderRadius: '5px'
                  }}>
                    <p>Orbit: Left-click drag</p>
                    <p>Pan: Right-click drag</p>
                    <p>Zoom: Scroll wheel</p>
                  </div>
                  
                  <AppInitializer onInitialized={handleInitialized} />
                </div>
                <Toaster />
              </SafeXRProvider>
            </TooltipProvider>
          </PanelProvider>
        </ApplicationModeProvider>
      </WindowSizeProvider>
    </ThemeProvider>
  );
};

export default ModernAppPage;