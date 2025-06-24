import { useEffect, Component, ReactNode, useCallback } from 'react'
import AppInitializer from './AppInitializer'
import { ThemeProvider } from '../design-system/ThemeProvider'
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext'
import { Toaster } from '../ui/Toaster'
// import { TooltipProvider } from '../ui/Tooltip'
import XRCoreProvider from '../features/xr/providers/XRCoreProvider'
// Removed GraphCanvas, ViewportContainer, MainLayout, DockingZone, ViewportControls, PanelProvider, Panel, SystemPanel, WindowSizeProvider
import { useSettingsStore } from '../store/settingsStore'
import { createLogger, createErrorMetadata } from '../utils/logger'
// Removed SimpleThreeWindowPage import as it's not used
// Removed SimpleThreeWindowPage import
// import SimpleGraphPage from '../pages/AppPage' // Corrected path: SimpleGraphPage is exported from AppPage.tsx
import TwoPaneLayout from './TwoPaneLayout'; // Added import for TwoPaneLayout
import { CommandPalette } from '../features/command-palette/components/CommandPalette'
import { initializeCommandPalette } from '../features/command-palette/defaultCommands'
import { HelpProvider } from '../features/help/components/HelpProvider'
import { registerSettingsHelp } from '../features/help/settingsHelp'
import { TooltipProvider } from '../ui/Tooltip'
import { OnboardingProvider } from '../features/onboarding/components/OnboardingProvider'
import { welcomeFlow, settingsFlow, advancedFlow, registerOnboardingCommands } from '../features/onboarding/flows/defaultFlows'

import '../styles/tokens.css'
// Removed layout.css import
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
  // Removed isLoading, panel visibility states, isSimpleMode state
  // Select the primitive value directly to avoid unnecessary re-renders
  const initialized = useSettingsStore(state => state.initialized)

  // Simplified useEffect, only checking initialization
  useEffect(() => {
    // Initialize command palette, help system, and onboarding on first load
    if (initialized) {
      initializeCommandPalette();
      registerSettingsHelp();
      registerOnboardingCommands();

      // Check if this is the first visit
      const hasVisited = localStorage.getItem('hasVisited');
      if (!hasVisited) {
        localStorage.setItem('hasVisited', 'true');
        // Start welcome tour after a short delay
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('start-onboarding', {
            detail: { flowId: 'welcome' }
          }));
        }, 1000);
      }
    }
  }, [initialized])

  // Wrap handleInitialized in useCallback to stabilize its reference
  const handleInitialized = useCallback(() => {
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('Application initialized');
    }
    // No need for setIsLoading(false) here if SimpleGraphPage handles its own loading
  }, []) // Dependency array is empty as it only uses getState

  // Removed viewport control handlers (handleResetCamera, etc.) as they belong in SimpleGraphPage or its children
  // Removed panel toggle handlers (handleToggleLeftPanel, etc.)
  // Removed handleViewportResize callback

  // No longer need the isSimpleMode check, always render SimpleGraphPage

  return (
    <ThemeProvider defaultTheme="dark">
      <TooltipProvider>
        <HelpProvider>
          <OnboardingProvider>
            {/* Removed WindowSizeProvider */}
            <ErrorBoundary>
              <ApplicationModeProvider>
                {/* Removed PanelProvider */}
                <XRCoreProvider>
                  {/* Render TwoPaneLayout only after settings are initialized */}
                  {initialized ? <TwoPaneLayout /> : <div>Loading application...</div>}
                  {!initialized && <AppInitializer onInitialized={handleInitialized} /> }
                  {/* Command Palette */}
                  <CommandPalette />
                  {/* Toaster remains at the top level */}
                  <Toaster />
                </XRCoreProvider>
              </ApplicationModeProvider>
            </ErrorBoundary>
          </OnboardingProvider>
        </HelpProvider>
      </TooltipProvider>
    </ThemeProvider>
  )
}

export default App
