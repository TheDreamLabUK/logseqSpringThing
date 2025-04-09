import React, { useEffect, useState, useCallback } from 'react';
import { useSafeXR, withSafeXR } from '../hooks/useSafeXRHooks';
import { MetadataVisualizer, useTextLabelManager } from '../../visualization/components/MetadataVisualizer';
import { useHandTracking } from '../systems/HandInteractionSystem';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('XRVisualizationConnector');

/**
 * XRVisualizationConnector connects the XR hand interaction system
 * with the visualization system and platform manager.
 * 
 * This component acts as the dependency injector between these systems.
 * It is wrapped with the XR context safety check to prevent errors.
 */
const XRVisualizationConnectorInner: React.FC = () => {
  const { isPresenting: isXRMode } = useSafeXR();
  const settings = useSettingsStore(state => state.settings);
  const handTracking = useHandTracking();
  const labelManager = useTextLabelManager();
  const [interactionEnabled, setInteractionEnabled] = useState(true);
  
  // Handle platform changes
  useEffect(() => {
    // Configure interactivity based on XR mode
    setInteractionEnabled(isXRMode && settings?.xr?.handTracking !== false); // Use correct property name
    
    // Debug logging
    if (isXRMode) {
      logger.info('XR mode active, configuring visualization for hand interaction');
    }
  }, [isXRMode, settings?.xr?.handTracking]); // Use correct property name
  
  // Handle hand gesture interactions with visualizations
  useEffect(() => {
    if (!interactionEnabled) return;
    
    // Example: Use pinch gesture state to interact with labels
    const { pinchState, handPositions, isLeftHandVisible, isRightHandVisible } = handTracking;
    
    // Update visualization system based on hand state using tuple positions
    // This is just a stub - real implementation would have more logic
    if (pinchState.left || pinchState.right) {
      // Use tuple based positions for hand interactions
      const leftPos = handPositions.left;
      const rightPos = handPositions.right;
      
      if (pinchState.left && leftPos) {
        logger.debug(`Left hand pinch at [${leftPos[0]}, ${leftPos[1]}, ${leftPos[2]}]`);
      }
      
      if (pinchState.right && rightPos) {
        logger.debug(`Right hand pinch at [${rightPos[0]}, ${rightPos[1]}, ${rightPos[2]}]`);
      }
    }
    
    return () => {
      // Cleanup if needed
    };
  }, [handTracking.pinchState, handTracking.handPositions, interactionEnabled]);
  
  // Render the visualization system with the appropriate settings
  return (
    <MetadataVisualizer 
      renderLabels={settings?.visualization?.labels?.enableLabels !== false} // Use correct property name
      // renderIcons={settings?.visualization?.icons?.enabled !== false} // Property 'icons' does not exist
      // renderMetrics={settings?.visualization?.metrics?.enabled} // Property 'metrics' does not exist
    />
  );
};

// Wrap with XR context safety check to prevent outside-XR-context errors
const XRVisualizationConnector = withSafeXR(XRVisualizationConnectorInner, 'XRVisualizationConnector');
export default XRVisualizationConnector;