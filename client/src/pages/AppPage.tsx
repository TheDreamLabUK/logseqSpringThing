import React, { Suspense, useEffect, useState, useCallback } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
// Import postprocessing effects
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import GraphManager from '../features/graph/components/GraphManager';
import ViewportControls from '../features/visualization/components/ViewportControls';
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext';
import { ControlPanelProvider } from '../features/settings/components/control-panel-context';
import { createLogger } from '../utils/logger';
import AppInitializer from '../app/AppInitializer';
import { useSettingsStore } from '../store/settingsStore';
import { ThemeProvider } from '../ui/ThemeProvider';
import { TooltipProvider } from '../ui/Tooltip';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/Collapsible'; // Added for UI
import NostrAuthSection from '../features/auth/components/NostrAuthSection'; // Added Auth
import SystemPanel from '../features/settings/components/panels/SystemPanel'; // Added Settings
import VisualizationPanel from '../features/settings/components/panels/VisualizationPanel'; // Added Settings
import XRPanel from '../features/settings/components/panels/XRPanel'; // Added Settings
import { HologramVisualization } from '../features/visualization/components/HologramVisualization'; // Added Hologram
import CameraController from '../features/visualization/components/CameraController'; // Import extracted component
// Removed WindowSizeProvider, MainLayout, PanelProvider imports
// Removed layout.css, globals.css, tokens.css imports

const logger = createLogger('SimpleGraphPage');

// Removed inline CameraController definition

const AppPage: React.FC = () => { // Renamed component
  // State variables
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50);
  const [sidebarVisible, setSidebarVisible] = useState(true);

  // Function to ensure sidebar toggle button is always visible
  const toggleSidebar = useCallback(() => {
    setSidebarVisible(prev => !prev);
    logger.debug(`Sidebar visibility toggled to: ${!sidebarVisible}`);
  }, [sidebarVisible]);

  // Fetch initial graph data
  useEffect(() => {
    const initializeGraph = async () => {
      try {
        // Removed console.log
        await graphDataManager.fetchInitialData();
        // Removed console.log
        const data = graphDataManager.getGraphData();
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        data.nodes.forEach((node) => {
          if (node.position) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
            minZ = Math.min(minZ, node.position.z);
            maxZ = Math.max(maxZ, node.position.z);
          }
        });
        const centerX = (maxX + minX) / 2 || 0;
        const centerY = (maxY + minY) / 2 || 0;
        const centerZ = (maxZ + minZ) / 2 || 0;
        const width = maxX - minX || 1;
        const height = maxY - minY || 1;
        const depth = maxZ - minZ || 1;
        const maxDimension = Math.max(width, height, depth);
        setGraphCenter([centerX, centerY, centerZ]);
        setGraphSize(maxDimension > 0 ? maxDimension : 50);
        setIsLoading(false);
      } catch (err) {
        console.error('SimpleGraphPage: Failed to fetch initial graph data:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred during data fetch.');
        setIsLoading(false);
      }
    };
    initializeGraph();
  }, []);

  // Viewport control handlers (kept for ViewportControls)
  const handleResetCamera = useCallback(() => { logger.debug('Reset camera'); }, []);
  const handleZoomIn = useCallback(() => { logger.debug('Zoom in'); }, []);
  const handleZoomOut = useCallback(() => { logger.debug('Zoom out'); }, []);
  const handleToggleFullscreen = useCallback(() => {
    logger.debug('Toggle fullscreen');
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => logger.error(`Fullscreen error: ${err.message}`));
    } else if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }, []);
  const handleRotateView = useCallback(() => { logger.debug('Rotate view'); }, []);

  // Removed unused handleToggleSidebar function

  // Callback for AppInitializer
  const handleInitialized = useCallback(() => {
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('App Initialized (AppPage context)'); // Updated log context
    }
  }, []);

  const backgroundColor = '#000000'; // Black background

  // Render content based on state
  const renderContent = () => {
    if (isLoading) {
      return <div style={{ padding: '2rem', color: 'white', backgroundColor: '#222' }}>Loading graph data...</div>;
    }
    if (error) {
      return <div style={{ padding: '2rem', color: 'red', backgroundColor: '#222' }}>Error loading graph data: {error}</div>;
    }
    // Simplified structure: Canvas container + ViewportControls + New Controls
    return (
      <div className="flex w-full h-full" style={{ height: '100vh' }}>
        {/* Main Canvas Container */}
        <div className="flex-grow relative" style={{ height: '100vh' }}>
          <Canvas
            className="three-canvas"
            style={{ display: 'block', width: '100%', height: '100%' }}
            camera={{ position: [0, 10, 50], fov: 75, near: 0.1, far: 2000 }}
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
              makeDefault
              enableDamping
              dampingFactor={0.05}
              minDistance={1}
              maxDistance={2000}
              target={graphCenter}
            />
            <Suspense fallback={null}>
              <GraphManager />
              <HologramVisualization standalone={false} position={[0, 0, 0]} size={20} />
            </Suspense>
            <axesHelper args={[2]} />
            <Stats />

            {/* Add bloom effect */}
            <EffectComposer>
              <Bloom
                luminanceThreshold={0.2}
                luminanceSmoothing={0.9}
                intensity={1.5}
              />
            </EffectComposer>
          </Canvas>

          {/* ViewportControls */}
          <ViewportControls
            className="absolute top-2 left-2 z-10"
            onReset={handleResetCamera}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onToggleFullscreen={handleToggleFullscreen}
            onRotate={handleRotateView} // Correctly passing the handler now
          />

          {/* Sidebar toggle button */}
          <button
            onClick={toggleSidebar}
            className="fixed top-4 right-4 z-[3000] inline-flex items-center justify-center rounded-md text-sm font-medium h-12 w-12 bg-primary text-primary-foreground shadow-lg"
            aria-label={sidebarVisible ? "Hide Sidebar" : "Show Sidebar"}
            style={{ boxShadow: '0 0 10px rgba(0,0,0,0.5)', border: '2px solid white' }}
          >
            {sidebarVisible ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                <path d="M19 12H5"/>
                <path d="M12 19l-7-7 7-7"/>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                <path d="M5 12h14"/>
                <path d="M12 5l7 7-7 7"/>
              </svg>
            )}
          </button>
        </div>

        {/* Right sidebar - Enhanced Styling */}
        {sidebarVisible && (
          <div className="w-80 h-screen flex flex-col border-l border-border bg-background z-[2000]"> {/* Use h-screen, flex column, increased z-index */}
            <div className="flex-1 p-4 space-y-6 overflow-y-auto"> {/* Add more vertical space (space-y-6), allow content scroll */}
              {/* Authentication Section */}
              <Collapsible defaultOpen={true} className="border rounded-lg p-3">
                <CollapsibleTrigger className="flex justify-between items-center text-lg font-semibold w-full text-left hover:bg-muted rounded p-2">
                  Authentication
                  {/* Add a chevron icon or similar indicator if available */}
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-3 px-2"> {/* Add padding top */}
                  <NostrAuthSection />
                </CollapsibleContent>
              </Collapsible>

              {/* Settings Section */}
              <Collapsible defaultOpen={true} className="border rounded-lg p-3">
                <CollapsibleTrigger className="flex justify-between items-center text-lg font-semibold w-full text-left hover:bg-muted rounded p-2">
                  Settings
                  {/* Add a chevron icon or similar indicator if available */}
                </CollapsibleTrigger>
                <CollapsibleContent className="pt-3 px-2 space-y-4"> {/* Add padding top and space between panels */}
                  {/* Wrap each panel for potential future styling/grouping */}
                  <div>
                    <h4 className="text-md font-medium mb-2 text-muted-foreground">System</h4>
                    <SystemPanel panelId="main-settings-system" />
                  </div>
                  <hr className="my-3 border-border" /> {/* Divider */}
                  <div>
                    <h4 className="text-md font-medium mb-2 text-muted-foreground">Visualization</h4>
                    <VisualizationPanel />
                  </div>
                  <hr className="my-3 border-border" /> {/* Divider */}
                  <div>
                    <h4 className="text-md font-medium mb-2 text-muted-foreground">XR</h4>
                    <XRPanel panelId="main-settings-xr" />
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Main component return with essential providers
  return (
    <ThemeProvider defaultTheme="dark">
      <TooltipProvider>
        <ApplicationModeProvider>
          <ControlPanelProvider>
            {renderContent()}
            <AppInitializer onInitialized={handleInitialized} />
          </ControlPanelProvider>
        </ApplicationModeProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
};

export default AppPage; // Updated export
