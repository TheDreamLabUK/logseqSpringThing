import React, { useEffect, useState, Component, ReactNode, useCallback } from 'react'
import AppInitializer from './AppInitializer'
import { ThemeProvider } from '../ui/ThemeProvider'
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext'
import { Toaster } from '../ui/Toaster'
import { TooltipProvider } from '../ui/Tooltip'
import GraphCanvas from '../features/graph/components/GraphCanvas'
import ViewportContainer from '../components/layout/ViewportContainer'
import SafeXRProvider from '../features/xr/providers/SafeXRProvider'
import MainLayout from '../components/layout/MainLayout'
import DockingZone from '../features/panel/components/DockingZone'
import ViewportControls from '../features/visualization/components/ViewportControls'
import { PanelProvider } from '../features/panel/components/PanelContext'
import Panel from '../features/panel/components/Panel'
import SystemPanel from '../features/settings/components/panels/SystemPanel'
import { WindowSizeProvider } from '../contexts/WindowSizeContext'
import { useSettingsStore } from '../store/settingsStore'
import { createLogger, createErrorMetadata } from '../utils/logger'
import '../styles/tokens.css'
import '../styles/layout.css'

const logger = createLogger('App')

// Error boundary component to catch rendering errors
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, { hasError: boolean; error: Error | null; errorInfo: any }> {
  state = { hasError: false, error: null, errorInfo: null };  
  
  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }
  
  componentDidCatch(error: any, errorInfo: any) {
    logger.error('React error boundary caught error:', {
      ...createErrorMetadata(error),
      component: errorInfo?.componentStack 
        ? errorInfo.componentStack.split('\n')[1]?.trim() 
        : 'Unknown component'
    });
    this.setState({ errorInfo });
  }
  
  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 bg-destructive text-destructive-foreground rounded-md">
          <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
          <p className="mb-4">The application encountered an error. Try refreshing the page.</p>
          {process.env.NODE_ENV === 'development' && (
            <pre className="bg-muted p-2 rounded text-sm overflow-auto">
              {this.state.error 
                ? (this.state.error.message || String(this.state.error)) 
                : 'No error details available'}
            </pre>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [isLoading, setIsLoading] = useState(true)
  const [showLeftPanel, setShowLeftPanel] = useState(false)
  const [showTopPanel, setShowTopPanel] = useState(true)
  const [showRightPanel, setShowRightPanel] = useState(true)
  const [topPanelDense, setTopPanelDense] = useState(true)
  // Select the primitive value directly to avoid unnecessary re-renders
  const initialized = useSettingsStore(state => state.initialized)

  useEffect(() => {
    if (initialized) {
      setIsLoading(false)
    }
  }, [initialized])

  // Wrap handleInitialized in useCallback to stabilize its reference
  const handleInitialized = useCallback(() => {
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('Application initialized');
    }
    setIsLoading(false)
  }, []) // Dependency array is empty as it only uses setIsLoading and getState

  // Viewport control handlers wrapped in useCallback
  const handleResetCamera = useCallback(() => {
    logger.debug('Reset camera')
  }, [])
  
  const handleZoomIn = useCallback(() => {
    logger.debug('Zoom in')
  }, [])
  
  const handleZoomOut = useCallback(() => {
    logger.debug('Zoom out')
  }, [])
  
  const handleToggleFullscreen = useCallback(() => {
    logger.debug('Toggle fullscreen')
  }, [])
  
  const handleRotateView = useCallback(() => {
    logger.debug('Rotate view')
  }, [])
  
  const handleToggleLeftPanel = useCallback(() => {
    setShowLeftPanel(prev => !prev)
  }, [])
  
  const handleToggleRightPanel = useCallback(() => {
    setShowRightPanel(prev => !prev)
  }, [])
  
  const handleToggleTopPanel = useCallback(() => {
    setShowTopPanel(prev => !prev)
  }, [])

  // Define stable onResize callback for ViewportContainer
  const handleViewportResize = useCallback((width: number, height: number) => {
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug(`ViewportContainer resized: ${Math.round(width)}×${Math.round(height)}`);
    }
  }, []); // Empty dependency array as getState is stable

  return (
    <ThemeProvider defaultTheme="dark">
      <WindowSizeProvider>
        <ErrorBoundary>
          <ApplicationModeProvider>
            <PanelProvider>
              <TooltipProvider>
                <SafeXRProvider>
                  <div 
                    className="app-container"
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      width: '100%',
                      height: '100%',
                      minHeight: '0',
                      overflow: 'hidden'
                    }}
                  >
                    <MainLayout
                      viewportContent={
                        // Pass the stable callback function as the prop
                        // Render GraphCanvas directly inside ViewportContainer
                        <ViewportContainer onResize={handleViewportResize}>
                          {!isLoading && (
                            <ErrorBoundary fallback={
                              <div className="flex items-center justify-center h-full">
                                <div className="p-4 bg-destructive/20 text-destructive-foreground rounded-md max-w-md">
                                  <h2 className="text-xl font-bold mb-2">Visualization Error</h2>
                                  <p>The 3D visualization component could not be loaded.</p>
                                  <p className="text-sm mt-2">This may be due to WebGL compatibility issues or problems with the graph data.</p>
                                </div>
                              </div>
                            }>
                              <GraphCanvas />
                            </ErrorBoundary>
                          )}
                        </ViewportContainer>
                      }
                      // Render overlays outside ViewportContainer, positioned relative to MainLayout's content area
                      overlays={
                        <>
                          {/* Loading Overlay - Positioned absolutely */}
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

                          {/* Viewport Controls Overlay - Positioned absolutely/fixed */}
                          {!isLoading && (
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
                          )}
                        </>
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
                    <AppInitializer onInitialized={handleInitialized} />
                  </div>
                  {/* Ensure Toaster is rendered outside potentially looping structures if possible */}
                  <Toaster />
                </SafeXRProvider>
              </TooltipProvider>
            </PanelProvider>
          </ApplicationModeProvider>
        </ErrorBoundary>
      </WindowSizeProvider>
    </ThemeProvider>
  )
}

export default App
