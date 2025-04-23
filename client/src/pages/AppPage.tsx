import React, { Suspense, useEffect, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
// Import postprocessing effects
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import GraphManager from '../features/graph/components/GraphManager';
// ViewportControls removed
import { createLogger } from '../utils/logger';
import { useSettingsStore } from '../store/settingsStore';
import { ThemeProvider } from '../ui/ThemeProvider';
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext';
import { ControlPanelProvider } from '../features/settings/components/control-panel-context';
import AppInitializer from '../app/AppInitializer';
// Removed unused Collapsible imports
// Removed panel/tab component imports - now handled by LowerControlPanel
import { HologramVisualization } from '../features/visualization/components/HologramVisualization';
import CameraController from '../features/visualization/components/CameraController';
// Removed icon imports - now handled by LowerControlPanel
// Import type definitions to fix JSX element errors
import '../types/react-three-fiber.d.ts';
// Ensure Three.js types are properly loaded
import * as THREE from 'three';
import LowerControlPanel from '../components/layout/LowerControlPanel'; // Import the new component

const logger = createLogger('SimpleGraphPage');

// Removed inline CameraController definition

const AppPage: React.FC = () => { // Renamed component
  console.log('AppPage rendering');

  // State variables
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50);

  // Settings panel is now always visible, no toggle needed

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

  // Viewport control handlers removed

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

    // Structure for fixed canvas and scrollable panel
    return (
      // Main container: Full height, flex column, overflow hidden to prevent whole page scroll
      <div className="flex flex-col w-full h-screen overflow-hidden bg-gray-900 text-white" style={{ backgroundColor: '#111827', color: 'white' }}>
        {/* Main Canvas Container - Fixed height */}
        <div className="relative flex-shrink-0" style={{ height: '70vh' }}> {/* Reduced height to give more space to settings */}
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

          {/* ViewportControls removed */}

          {/* Toggle button removed - settings panel is always visible */}
        </div>

        {/* Lower Panel Container - Takes remaining space (flex-1) and handles overall scrolling */}
        <div className="flex-1 w-full border-t border-gray-700 overflow-y-auto bg-gray-900 text-white" style={{ backgroundColor: '#111827', color: 'white', overflowY: 'auto' }}> {/* Using direct color classes and inline styles */}
          {/* Render the new LowerControlPanel component */}
          <LowerControlPanel />
        </div>
      </div>
    );
  };

  // Main component return with essential providers
  return (
    <ThemeProvider defaultTheme="dark">
      <ApplicationModeProvider>
        <ControlPanelProvider>
          {renderContent()}
          <AppInitializer onInitialized={handleInitialized} />
        </ControlPanelProvider>
      </ApplicationModeProvider>
    </ThemeProvider>
  );
};

export default AppPage; // Updated export
