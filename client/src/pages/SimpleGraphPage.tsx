import React, { Suspense, useEffect, useState, useCallback } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import GraphManager from '../features/graph/components/GraphManager';
import ViewportControls from '../features/visualization/components/ViewportControls';
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext';
import { createLogger } from '../utils/logger';
import AppInitializer from '../app/AppInitializer';
import { useSettingsStore } from '../store/settingsStore';
import { ThemeProvider } from '../ui/ThemeProvider';
import { TooltipProvider } from '../ui/Tooltip';
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
  const handleToggleLeftPanel = useCallback(() => {}, []);
  const handleToggleRightPanel = useCallback(() => {}, []);
  const handleToggleTopPanel = useCallback(() => {}, []);

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
      <div className="relative w-full h-full"> {/* Basic container */}
        <div className="absolute inset-0"> {/* Canvas Container */}
          <Canvas
            className="three-canvas"
            style={{ display: 'block' }}
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
            </Suspense>
            <axesHelper args={[2]} />
            <Stats />
          </Canvas>
        </div> {/* End Canvas Container */}

        {/* ViewportControls rendered as a sibling, positioning might be off */}
        <ViewportControls
          onReset={handleResetCamera}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onToggleFullscreen={handleToggleFullscreen}
          onRotate={handleRotateView}
          onToggleLeftPanel={handleToggleLeftPanel}
          onToggleRightPanel={handleToggleRightPanel}
          onToggleTopPanel={handleToggleTopPanel}
        />

        {/* New Simple Controls Placeholder */}
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          zIndex: 1001, // Ensure it's above canvas/other elements if needed
          background: 'rgba(255, 255, 255, 0.7)',
          padding: '5px',
          borderRadius: '4px',
          display: 'flex',
          gap: '5px'
        }}>
          <button>Zoom In</button>
          <button>Zoom Out</button>
          <button>Reset</button>
          <button>Full</button>
        </div>

      </div> // End Basic container
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
