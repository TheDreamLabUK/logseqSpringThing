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
import NostrAuthSection from '../features/auth/components/NostrAuthSection';
import SystemPanel from '../features/settings/components/panels/SystemPanel';
import VisualizationPanel from '../features/settings/components/panels/VisualizationPanel';
import XRPanel from '../features/settings/components/panels/XRPanel';
import AIPanel from '../features/settings/components/panels/AIPanel'; // Import AIPanel
import Tabs from '../ui/Tabs'; // Import Tabs component
import { HologramVisualization } from '../features/visualization/components/HologramVisualization';
import CameraController from '../features/visualization/components/CameraController';
// Import icons for tabs (using available lucide-react icons)
// Temporarily using only Settings icon to debug import issues
import { Settings, Eye } from 'lucide-react';

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
      <div className="flex flex-col w-full h-screen overflow-hidden bg-background text-foreground">
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

        {/* Settings Panel Container - Takes remaining space and scrolls internally */}
          <div className="flex-1 w-full bg-background text-foreground border-t border-gray-800 overflow-y-auto"> {/* Using theme variables */}
            <div className="container mx-auto px-4 py-6"> {/* Reduced padding */}
              {/* Header with gradient underline */}
              <div className="mb-6 text-center"> {/* Reduced margin */}
                <h2 className="text-3xl font-bold mb-2">Control Panel</h2>
                <div className="h-1 w-24 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto"></div>
              </div>

              {/* Tabbed Settings Area */}
              <div className="bg-card rounded-lg overflow-hidden shadow-xl border border-border min-h-[300px]"> {/* Using theme variables */}
                <Tabs
                  tabs={[
                    {
                      label: 'Auth',
                      icon: <Settings className="h-4 w-4" />, // Temp: Use Settings
                      content: <NostrAuthSection />,
                    },
                    {
                      label: 'System',
                      icon: <Settings className="h-4 w-4" />,
                      content: <SystemPanel panelId="main-settings-system" />,
                    },
                    {
                      label: 'Visualization',
                      icon: <Eye className="h-4 w-4" />, // Keep Eye as it seems okay
                      content: <VisualizationPanel />,
                    },
                    {
                      label: 'XR',
                      icon: <Settings className="h-4 w-4" />, // Temp: Use Settings
                      content: <XRPanel panelId="main-settings-xr" />,
                    },
                    {
                      label: 'AI Services',
                      icon: <Settings className="h-4 w-4" />, // Temp: Use Settings
                      content: <AIPanel />,
                    },
                  ]}
                  tabListClassName="bg-card px-4" // Use theme variables
                  tabButtonClassName="py-3" // Adjust button padding
                  tabContentClassName="bg-card text-card-foreground" // Ensure content area matches theme
                />
              </div> {/* Correct placement for the closing div of the tabbed area */}

              {/* Footer with version info */}
              <div className="mt-12 text-center text-gray-500 text-sm">
                <p>LogseqSpringThing v0.1.0 | Made with ❤️ by the community</p>
              </div>
            </div>
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
