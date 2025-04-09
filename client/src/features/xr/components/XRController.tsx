import React, { useState, useCallback } from 'react'
import { useSafeXR, withSafeXR } from '../hooks/useSafeXRHooks'
import HandInteractionSystem, { GestureRecognitionResult } from '../systems/HandInteractionSystem'
import { debugState } from '../../../utils/debugState'
import { useSettingsStore } from '../../../store/settingsStore'
import { createLogger } from '../../../utils/logger'

const logger = createLogger('XRController')

/**
 * XRControllerInner component handles WebXR functionality through react-three/xr.
 * XRController component manages WebXR functionality through react-three/xr.
 * This version is simplified to avoid integration conflicts.
 */
const XRController: React.FC = () => {
  const { isPresenting, controllers } = useSafeXR()
  const settings = useSettingsStore(state => state.settings)
  const [handsVisible, setHandsVisible] = useState(false)
  const [handTrackingEnabled, setHandTrackingEnabled] = useState(settings?.xr?.handTracking !== false) // Use correct property name from settings.ts
  
  // Log session state changes
  React.useEffect(() => {
    if (debugState.isEnabled()) {
      if (isPresenting) {
        logger.info('XR session is now active')
      } else {
        logger.info('XR session is not active')
      }
    }
  }, [isPresenting])

  // Log controller information
  React.useEffect(() => {
    if (isPresenting && controllers && controllers.length > 0 && debugState.isEnabled()) {
      logger.info(`XR controllers active: ${controllers.length}`)
      controllers.forEach((controller, index) => {
        logger.info(`Controller ${index}: ${controller.inputSource.handedness}`)
      })
    }
  }, [controllers, isPresenting])

  // Handle gesture recognition
  const handleGestureRecognized = useCallback((gesture: GestureRecognitionResult) => {
    if (debugState.isEnabled()) {
      logger.info(`Gesture recognized: ${gesture.gesture} (${gesture.confidence.toFixed(2)}) with ${gesture.hand} hand`)
    }
  }, [])

  // Handle hand visibility changes
  const handleHandsVisible = useCallback((visible: boolean) => {
    setHandsVisible(visible)
    
    if (debugState.isEnabled()) {
      logger.info(`Hands visible: ${visible}`)
    }
  }, [])
  
  // Only render if enabled in settings
  if (settings?.xr?.enabled === false) {
    return null
  }
  
  return (
    <group name="xr-controller-root">
      <HandInteractionSystem 
        enabled={handTrackingEnabled}
        onGestureRecognized={handleGestureRecognized}
        onHandsVisible={handleHandsVisible}
      />
    </group>
  )
}

// Wrap with XR context safety check to prevent outside-XR-context errors
const SafeXRController = withSafeXR(XRController, 'XRController');
export default SafeXRController