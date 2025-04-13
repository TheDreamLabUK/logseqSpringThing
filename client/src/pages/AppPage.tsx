import React, { Suspense, useEffect, useState, useCallback } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import GraphManager from '../features/graph/components/GraphManager';
import ViewportControls from '../features/visualization/components/ViewportControls';
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext';
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
// Removed WindowSizeProvider, MainLayout, PanelProvider imports
// Removed layout.css, globals.css, tokens.css imports

const logger = createLogger('SimpleGraphPage');

// Camera controller
const CameraController = ({ center, size }: { center: [number, number, number], size: number }) => {
  const { camera } = useThree();
  useEffect(() => {
    console.log(`Setting camera to look at center: [${center}] with distance: ${size*1.5}`);
    camera.position.set(center[0], center[1], center[2] + size*1.5);
    camera.lookAt(center[0], center[1], center[2]);
    camera.updateProjectionMatrix();
  }, [camera, center, size]);
  return null;
};

const SimpleGraphPage: React.FC = () => {
  // State variables
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50);
  const [sidebarVisible, setSidebarVisible] = useState(true);

  // Fetch initial graph data
  useEffect(() => {
    const initializeGraph = async () => {
      try {
        console.log('SimpleGraphPage: Fetching initial graph data...');
        await graphDataManager.fetchInitialData();
        console.log('SimpleGraphPage: Initial graph data fetched successfully.');
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

  // Toggle sidebar visibility
  const handleToggleSidebar = useCallback(() => {
    setSidebarVisible(prev => !prev);
    logger.debug('Toggle sidebar');
  }, []);

  // Callback for AppInitializer
  const handleInitialized = useCallback(() => {
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('App Initialized (SimpleGraphPage context)');
    }
  }, []);

  const blueBackgroundColor = '#0000cd';

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
      // Main container using Flexbox for side-by-side layout
      <div className="flex w-full h-full overflow-hidden" style={{ height: '100vh' }}>

        {/* Canvas Container - takes most of the space */}
        <div className="flex-grow h-full relative" style={{ height: '100vh' }}>
          <Canvas
            className="three-canvas"
            style={{ display: 'block', width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }} // Ensure canvas fills container
            camera={{ position: [0, 0, 50], fov: 50, near: 0.1, far: 2000 }}
            gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
            dpr={[1, 2]}
            shadows
          >
            <color attach="background" args={[blueBackgroundColor]} />
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
              {/* Add the 3D Hologram into the scene */}
              <HologramVisualization standalone={false} position={[0, 0, -20]} size={10} />
            </Suspense>
            <axesHelper args={[2]} />
            <Stats />
          </Canvas>

          {/* ViewportControls positioned absolutely INSIDE the relative canvas container */}
          <ViewportControls
            className="absolute top-2 left-2 z-10" // Positioning relative to the div above
            onReset={handleResetCamera}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onToggleFullscreen={handleToggleFullscreen}
            onRotate={handleRotateView}
          />

          {/* Sidebar toggle button */}
          <button
            onClick={handleToggleSidebar}
            className="absolute top-2 right-2 z-10 inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground h-8 w-8"
            aria-label={sidebarVisible ? "Hide Sidebar" : "Show Sidebar"}
          >
            {sidebarVisible ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
                <path d="M19 12H5"/>
                <path d="M12 19l-7-7 7-7"/>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
                <path d="M5 12h14"/>
                <path d="M12 5l7 7-7 7"/>
              </svg>
            )}
          </button>
        </div> {/* End Canvas Container */}

        {/* Right sidebar for controls - collapsible */}
        {sidebarVisible && (
          <div className="flex-shrink-0 w-80 h-full border-l border-border bg-background overflow-y-auto" style={{ height: '100vh' }}>
            <div className="p-4 space-y-4">
              <Collapsible defaultOpen={true}>
                <CollapsibleTrigger className="text-lg font-semibold w-full text-left p-1 hover:bg-muted rounded">Authentication</CollapsibleTrigger>
                <CollapsibleContent className="p-2 border rounded mt-1">
                  <NostrAuthSection />
                </CollapsibleContent>
              </Collapsible>

              <Collapsible defaultOpen={true}>
                <CollapsibleTrigger className="text-lg font-semibold w-full text-left p-1 hover:bg-muted rounded mt-2">Settings</CollapsibleTrigger>
                <CollapsibleContent className="p-2 border rounded mt-1 space-y-4">
                  {/* Pass a dummy panelId, might need adjustment if component relies heavily on it */}
                  <SystemPanel panelId="main-settings-system" />
                  <VisualizationPanel /> {/* Removed panelId prop */}
                  <XRPanel panelId="main-settings-xr" />
                </CollapsibleContent>
              </Collapsible>

              {/* Add other feature components here as needed */}
            </div>
          </div>
        )} {/* End Right Sidebar */}

      </div> // End Main Flex container
    );
  };

  // Main component return with essential providers
  return (
    <ThemeProvider defaultTheme="dark">
      <TooltipProvider>
        <ApplicationModeProvider>
          {renderContent()}
          <AppInitializer onInitialized={handleInitialized} />
        </ApplicationModeProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
};

export default SimpleGraphPage;
