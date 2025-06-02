import React, { useEffect, useState, useCallback } from 'react';
import { useXRCore } from '../providers/XRCoreProvider';
import { MetadataVisualizer, useTextLabelManager } from '../../visualisation/components/MetadataVisualizer';
import { useHandTracking } from '../systems/HandInteractionSystem';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('XRVisualisationConnector');

/**
 * XRVisualisationConnector connects the Quest 3 AR hand interaction system
 * with the visualisation system and platform manager.
 *
 * This component acts as the dependency injector between these systems.
 * It uses the enhanced XRCoreProvider for robust session management.
 */
const XRVisualisationConnector: React.FC = () => {
  const { isSessionActive: isXRMode, sessionType } = useXRCore();
  const settings = useSettingsStore(state => state.settings);
  const handTracking = useHandTracking();
  const labelManager = useTextLabelManager();
  const [interactionEnabled, setInteractionEnabled] = useState(true);
  
  // Handle platform changes (Quest 3 AR focused)
  useEffect(() => {
    // Configure interactivity based on Quest 3 AR mode
    const isQuest3AR = isXRMode && sessionType === 'immersive-ar';
    setInteractionEnabled(isQuest3AR && settings?.xr?.enableHandTracking !== false);
    
    // Debug logging
    if (isQuest3AR) {
      if (settings?.system?.debug?.enabled) {
        logger.info('Quest 3 AR mode active, configuring visualisation for hand interaction');
      }
    }
  }, [isXRMode, sessionType, settings?.xr?.enableHandTracking, settings?.system?.debug?.enabled]);
  
  // Handle hand gesture interactions with visualisations
  useEffect(() => {
    if (!interactionEnabled) return;
    
    // Example: Use pinch gesture state to interact with labels
    const { pinchState, handPositions, isLeftHandVisible, isRightHandVisible } = handTracking;
    
    // Update visualisation system based on hand state using tuple positions
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
  
  // Render the visualisation system with the appropriate settings
  return (
    <MetadataVisualizer 
      renderLabels={settings?.visualisation?.labels?.enableLabels !== false} // Use correct property name
      // renderIcons={settings?.visualisation?.icons?.enabled !== false} // Property 'icons' does not exist
      // renderMetrics={settings?.visualisation?.metrics?.enabled} // Property 'metrics' does not exist
    />
  );
};

export default XRVisualisationConnector;