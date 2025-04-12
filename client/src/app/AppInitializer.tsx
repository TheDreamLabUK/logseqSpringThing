import React, { useEffect } from 'react';
import { createLogger, createErrorMetadata } from '../utils/logger';
import { debugState } from '../utils/debugState';
import { useSettingsStore } from '../store/settingsStore';
import WebSocketService from '../services/WebSocketService';
import { graphDataManager } from '../features/graph/managers/graphDataManager';

// Compatibility function to maintain compatibility with code expecting a loadServices function
const loadServices = async (): Promise<void> => {
  if (debugState.isEnabled()) {
    logger.info('Services pre-loaded via direct imports');
  }
}

const logger = createLogger('AppInitializer');

interface AppInitializerProps {
  onInitialized: () => void;
}

const AppInitializer: React.FC<AppInitializerProps> = ({ onInitialized }) => {
  const { settings, initialize } = useSettingsStore();

  useEffect(() => {
    const initApp = async () => {
      // Load services first
      await loadServices();
      
      if (debugState.isEnabled()) {
        logger.info('Starting application initialization...');
        }

        try {
          // Initialize settings
          const settings = await initialize();
  
          // Apply debug settings safely
          if (settings.system?.debug) {
            try {
              const debugSettings = settings.system.debug;
              debugState.enableDebug(debugSettings.enabled);
              if (debugSettings.enabled) {
                debugState.enableDataDebug(debugSettings.enableDataDebug);
                debugState.enablePerformanceDebug(debugSettings.enablePerformanceDebug);
              }
            } catch (debugError) {
              logger.warn('Error applying debug settings:', createErrorMetadata(debugError));
            }
          }

          // Try to initialize WebSocket
          if (typeof WebSocketService !== 'undefined' && typeof graphDataManager !== 'undefined') {
            try {
              // Initialize WebSocket
              await initializeWebSocket(settings); 
              // logger.info('WebSocket initialization deliberately disabled - using REST API only.'); // Commented out the disabling message
            } catch (wsError) {
              logger.error('WebSocket initialization failed, continuing with UI only:', createErrorMetadata(wsError));
              // We'll proceed without WebSocket connectivity
            }
          } else {
            logger.warn('WebSocket services not available, continuing with UI only');
          }
          
          // Fetch initial graph data AFTER settings and BEFORE signaling completion
          try {
            logger.info('Fetching initial graph data via REST API');
            await graphDataManager.fetchInitialData();
            if (debugState.isDataDebugEnabled()) {
              logger.debug('Initial graph data fetched successfully');
            }
          } catch (fetchError) {
            logger.error('Failed to fetch initial graph data:', createErrorMetadata(fetchError));
            // Initialize with empty data as fallback
            graphDataManager.setGraphData({ nodes: [], edges: [] });
          }
          
          if (debugState.isEnabled()) {
            logger.info('Application initialized successfully');
          }
        
          // Signal that initialization is complete
          onInitialized();
        
      } catch (error) {
          logger.error('Failed to initialize application components:', createErrorMetadata(error));
          // Even if initialization fails, try to signal completion to show UI
          onInitialized();
      }
    };

    initApp();
  }, [initialize, onInitialized]);

  // Initialize WebSocket and set up event handlers - now safer with more error handling
  const initializeWebSocket = async (settings: any): Promise<void> => {
    try {
      const websocketService = WebSocketService.getInstance();
      
      // Handle binary position updates from WebSocket
      websocketService.onBinaryMessage((data) => {
        if (data instanceof ArrayBuffer) {
          try {
            // Process binary position update through graph data manager
            graphDataManager.updateNodePositions(data);
            if (debugState.isDataDebugEnabled()) {
              logger.debug(`Processed binary position update: ${data.byteLength} bytes`);
            }
          } catch (error) {
            logger.error('Failed to process binary position update:', createErrorMetadata(error));
            
            // Add diagnostic info in debug mode
            if (debugState.isEnabled()) {
              // Display basic info about the data
              logger.debug(`Binary data size: ${data.byteLength} bytes`);
              
              // Display the first few bytes for debugging - helps detect compression headers
              try {
                const view = new DataView(data);
                const hexBytes = [];
                const maxBytesToShow = Math.min(16, data.byteLength);
                
                for (let i = 0; i < maxBytesToShow; i++) {
                  hexBytes.push(view.getUint8(i).toString(16).padStart(2, '0'));
                }
                
                logger.debug(`First ${maxBytesToShow} bytes: ${hexBytes.join(' ')}`);
                
                // Check if data might be compressed (zlib headers)
                if (data.byteLength >= 2) {
                  const firstByte = view.getUint8(0);
                  const secondByte = view.getUint8(1);
                  if (firstByte === 0x78 && (secondByte === 0x01 || secondByte === 0x9C || secondByte === 0xDA)) {
                    logger.debug('Data appears to be zlib compressed (has zlib header)');
                  }
                }
              } catch (e) {
                logger.debug('Could not display binary data preview');
              }
              
              // Check if the data length is a multiple of expected formats
              const nodeSize = 26; // 2 bytes (ID) + 12 bytes (position) + 12 bytes (velocity)
              if (data.byteLength % nodeSize !== 0) {
                logger.debug(`Invalid data length: not a multiple of ${nodeSize} bytes per node (remainder: ${data.byteLength % nodeSize})`);
              }
            }
          }
        }
      });
      
      // Set up connection status handler
      websocketService.onConnectionStatusChange((connected) => {
        if (debugState.isEnabled()) {
          logger.info(`WebSocket connection status changed: ${connected}`);
        }

        // Check if websocket is both connected AND ready (received 'connection_established' message)
        if (connected) {
          try {
            if (websocketService.isReady()) {
              // WebSocket is fully ready, now it's safe to enable binary updates
              logger.info('WebSocket is connected and fully established - enabling binary updates');
              graphDataManager.setBinaryUpdatesEnabled(true);
              if (debugState.isDataDebugEnabled()) {
                logger.debug('Binary updates enabled');
              }
            } else {
              logger.info('WebSocket connected but not fully established yet - waiting for readiness');
              
              // We'll let graphDataManager handle the binary updates enablement
              // through its retry mechanism that now checks for websocket readiness
              graphDataManager.enableBinaryUpdates();
            }
          } catch (connectionError) {
            logger.error('Error during WebSocket status change handling:', createErrorMetadata(connectionError));
          }
        }
      });
      
      // Configure GraphDataManager with WebSocket service (adapter pattern)
      if (websocketService) {
        const wsAdapter = {
          send: (data: ArrayBuffer) => {
            websocketService.sendRawBinaryData(data);
          },
          isReady: () => websocketService.isReady()
        };
        graphDataManager.setWebSocketService(wsAdapter);
      }
      
      try {
        // Connect WebSocket
        await websocketService.connect();
      } catch (connectError) {
        logger.error('Failed to connect to WebSocket:', createErrorMetadata(connectError));
      }
    } catch (error) {
      logger.error('Failed during WebSocket/data initialization:', createErrorMetadata(error));
      throw error;
    }
  };

  return null; // This component doesn't render anything directly
};

export default AppInitializer;
