import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three'; // Add this line
import { Group, Line, Raycaster, BufferGeometry, Vector3, Matrix4, LineBasicMaterial, Object3D } from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { Interactive } from '@react-three/xr';
import { usePlatform } from '../../../services/platformManager';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/logger';
import { GestureState, XRInteractionMode, InteractableObject } from '../types/xr';

const logger = createLogger('HandInteraction');

// Simplified XR handedness type
type XRHandedness = 'left' | 'right' | 'none';

// Interaction event types
type InteractionEventType = 'select' | 'hover' | 'unhover' | 'squeeze' | 'move';
type InteractionDistance = 'near' | 'far';
type InteractionEvent = { 
  type: InteractionEventType, 
  distance: InteractionDistance, 
  controller?: Object3D, 
  hand?: XRHandedness, 
  point?: [number, number, number] 
};

// Interface for recognized gesture
export interface GestureRecognitionResult {
  gesture: string;
  confidence: number;
  hand: XRHandedness;
}

// Props for the hand interaction system
interface HandInteractionSystemProps {
  children?: React.ReactNode;
  onGestureRecognized?: (gesture: GestureRecognitionResult) => void;
  onHandsVisible?: (visible: boolean) => void;
  enabled?: boolean;
  interactionMode?: XRInteractionMode;
  interactionDistance?: number;
  hapticFeedback?: boolean;
}

/**
 * Modern hand interaction system for WebXR
 * Uses React Three Fiber for Quest hand tracking
 */
export const HandInteractionSystem: React.FC<HandInteractionSystemProps> = ({
  children,
  onGestureRecognized,
  onHandsVisible,
  enabled = true,
  interactionMode = 'both',
  interactionDistance = 1.5,
  hapticFeedback = true
}) => {
  const { scene, gl, camera } = useThree();
  const { isPresenting, session, controllers, player } = useSafeXR();
  const platform = usePlatform();
  const settings = useSettingsStore(state => state.settings.xr);
  const handTrackingEnabled = settings.handTracking && enabled;
  
  // State for hands and interaction
  const [handsVisible, setHandsVisible] = useState(false);
  const [visualizeHands, setVisualizeHands] = useState(false);
  const [interactables, setInteractables] = useState<InteractableObject[]>([]);
  const [selectedObject, setSelectedObject] = useState<Object3D | null>(null);
  const [hoveredObject, setHoveredObject] = useState<Object3D | null>(null);

  // References for hand state
  const leftHandRef = useRef<Group | null>(null);
  const rightHandRef = useRef<Group | null>(null);
  const leftControllerRef = useRef<Group | null>(null);
  const rightControllerRef = useRef<Group | null>(null);
  const leftRayRef = useRef<Line | null>(null);
  const rightRayRef = useRef<Line | null>(null);
  
  // Gesture state reference
  const gestureStateRef = useRef<GestureState>({
    left: { pinch: false, grip: false, point: false, thumbsUp: false },
    right: { pinch: false, grip: false, point: false, thumbsUp: false }
  });

  // Raycaster for interaction
  const raycasterRef = useRef<Raycaster>(new Raycaster());
  
  // Initialize raycaster with proper settings
  useEffect(() => {
    if (raycasterRef.current) {
      raycasterRef.current.near = 0.01;
      raycasterRef.current.far = interactionDistance;
      (raycasterRef.current.params as any).Line = { threshold: 0.2 };
      (raycasterRef.current.params as any).Points = { threshold: 0.2 };
    }
  }, [interactionDistance]);

  // Collect all interactable objects in the scene
  useEffect(() => {
    // In a real implementation, this would scan the scene for objects with interactable components
    // For now, we'll just have an empty array that would be populated by components
  }, [scene]);
  
  // Map to store joint objects
  const jointsRef = useRef<Map<string, Object3D>>(new Map());

  // Initialize hand tracking
  useEffect(() => {
    if (!handTrackingEnabled) return;
    
    // Create hand groups if they don't exist
    if (!leftHandRef.current) {
      leftHandRef.current = new Group();
      (leftHandRef.current as any).name = 'left-hand';
      scene.add(leftHandRef.current);
    }

    // Create controller rays if they don't exist
    if (!leftRayRef.current) {
      const geometry = new BufferGeometry();
      // Manually create points instead of using Vector3 constructor
      const points = [
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: -interactionDistance }
      ];
      const rayGeometry = (geometry as any).setFromPoints(
        points.map(p => new (THREE as any).Vector3(p.x, p.y, p.z))
      );
      const rayMaterial = new (THREE as any).LineBasicMaterial({ 
        color: settings.controllerRayColor || 0x00ff00,
        opacity: 0.7, 
        transparent: true 
      });
      leftRayRef.current = new (THREE as any).Line(rayGeometry, rayMaterial);
    }
    
    if (!rightHandRef.current) {
      rightHandRef.current = new Group();
      (rightHandRef.current as any).name = 'right-hand';
      scene.add(rightHandRef.current);
    }
    
    // Create right controller ray
    if (!rightRayRef.current) {
      const geometry = new BufferGeometry();
      // Manually create points instead of using Vector3 constructor
      const points = [
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: -interactionDistance }
      ];
      const rayGeometry = (geometry as any).setFromPoints(
        points.map(p => new (THREE as any).Vector3(p.x, p.y, p.z))
      );
      const rayMaterial = new (THREE as any).LineBasicMaterial({ 
        color: settings.controllerRayColor || 0x00ff00,
        opacity: 0.7, 
        transparent: true 
      });
      rightRayRef.current = new (THREE as any).Line(rayGeometry, rayMaterial);
    }
    logger.info('Hand tracking system initialized');
    
    // Return cleanup function
    return () => {
      if (leftHandRef.current) {
        scene.remove(leftHandRef.current);
        leftHandRef.current = null;
      }
      
      if (rightHandRef.current) {
        scene.remove(rightHandRef.current);
        rightHandRef.current = null;
      }

      if (leftRayRef.current) {
        scene.remove(leftRayRef.current);
        leftRayRef.current = null;
      }
      
      if (rightRayRef.current) {
        scene.remove(rightHandRef.current);
        rightHandRef.current = null;
      }
      
      jointsRef.current.clear();
      logger.info('Hand tracking system disposed');
    };
  }, [handTrackingEnabled, scene, interactionDistance, settings.controllerRayColor]);
  
  // Update controller references when WebXR session changes
  useEffect(() => {
    if (!isPresenting || !platform.isWebXRSupported) return;
    
    // Attach to XR controllers if available
    if (controllers && controllers.length > 0) {
      controllers.forEach(controller => {
        if (controller.inputSource.handedness === 'left') {
          leftControllerRef.current = controller.controller;
          if (leftRayRef.current) {
            controller.controller.add(leftRayRef.current);
          }
        } else if (controller.inputSource.handedness === 'right') {
          rightControllerRef.current = controller.controller;
          if (rightRayRef.current) {
            controller.controller.add(rightRayRef.current);
          }
        }
      });
    }
    
    // Set up controller event listeners
    const handleControllerEvent = (event: any, type: InteractionEventType, hand: XRHandedness) => {
      handleInteractionEvent({
        type,
        distance: 'far',
        controller: hand === 'left' ? leftControllerRef.current as Object3D : rightControllerRef.current as Object3D,
        hand,
        point: event.intersections?.[0]?.point
      });
    };
    
    // Return cleanup function that removes event listeners
    return () => {
      // In a real implementation, we would remove event listeners here
    };
  }, [isPresenting, platform.isWebXRSupported, controllers, hapticFeedback]);
  
  // Handle various interaction events from controllers or hand tracking
  const handleInteractionEvent = (event: InteractionEvent) => {
    // Process different event types
    switch (event.type) {
      case 'select':
        // Handle selection (trigger press)
        if (hoveredObject) {
          setSelectedObject(hoveredObject);
          
          // Trigger haptic feedback if enabled
          if (hapticFeedback && event.controller && session) {
            const gamepad = (event.controller as any).inputSource?.gamepad;
            if (gamepad && gamepad.hapticActuators && gamepad.hapticActuators[0]) {
              gamepad.hapticActuators[0].pulse(0.5, 100);
            }
          }
          
          logger.info(`Selected object: ${hoveredObject.name}`);
        }
        break;
        
      case 'hover':
        // Handle hover state (ray pointing at object)
        if (event.point && event.controller) {
          setHoveredObject(event.controller);
          logger.debug(`Hovering object at ${event.point[0]}, ${event.point[1]}, ${event.point[2]}`);
        }
        break;
        
      case 'unhover':
        // Clear hover state
        setHoveredObject(null);
        break;
        
      case 'squeeze':
        // Handle grip button press
        if (selectedObject) {
          logger.info(`Squeezing object: ${selectedObject.name}`);
          
          // Trigger stronger haptic feedback for squeeze
          if (hapticFeedback && event.controller && session) {
            const gamepad = (event.controller as any).inputSource?.gamepad;
            if (gamepad && gamepad.hapticActuators && gamepad.hapticActuators[0]) {
              gamepad.hapticActuators[0].pulse(0.8, 150);
            }
          }
        }
        break;
        
      case 'move':
        // Handle movement of selected object
        if (selectedObject && event.controller) {
          // In a real implementation, this would update the position/rotation of the selected object
          logger.debug(`Moving selected object with controller`);
        }
        break;
    }
  };
  
  // Perform gesture recognition on hand joints
  const recognizeGestures = (handedness: XRHandedness, joints: Map<string, Object3D>) => {
    if (joints.size === 0) return;
    
    // Get key finger joints for gesture recognition
    const thumbTip = joints.get('thumb-tip');
    const indexTip = joints.get('index-finger-tip');
    const indexMiddle = joints.get('index-finger-phalanx-intermediate');
    const middleTip = joints.get('middle-finger-tip');
    const ringTip = joints.get('ring-finger-tip');
    const pinkyTip = joints.get('pinky-finger-tip');
    const wrist = joints.get('wrist');
    
    if (!thumbTip || !indexTip || !wrist) return;
    
    // Check thumb-index pinch
    // Calculate distance using tuple positions
    const thumbPos = [(thumbTip as any).position.x, (thumbTip as any).position.y, (thumbTip as any).position.z] as const;
    const indexPos = [(indexTip as any).position.x, (indexTip as any).position.y, (indexTip as any).position.z] as const;
    const dx = thumbPos[0] - indexPos[0];
    const dy = thumbPos[1] - indexPos[1];
    const dz = thumbPos[2] - indexPos[2];
    const thumbToIndexDistance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const isPinching = thumbToIndexDistance < 0.03; // 3cm threshold
    
    // Check point gesture (index extended, others curled)
    const isPointing = false; // Simplified - would check index extension and other fingers curled
    
    // Check grip gesture (all fingers curled)
    const isGripping = false; // Simplified - would check all fingers curled
    
    // Check thumbs up gesture
    const isThumbsUp = false; // Simplified - would check thumb orientation
    
    // Update gesture state
    const previousState = gestureStateRef.current[handedness];
    gestureStateRef.current[handedness] = {
      pinch: isPinching,
      grip: isGripping,
      point: isPointing,
      thumbsUp: isThumbsUp
    };
    
    // Notify about gesture changes
    if (isPinching && !previousState.pinch && onGestureRecognized) {
      onGestureRecognized({
        gesture: 'pinch',
        confidence: 0.9,
        hand: handedness
      });
      
      // Trigger interaction event for pinch
      handleInteractionEvent({ type: 'select', distance: 'near', hand: handedness });
    }
  };
  
  // Process hand data on each frame
  useFrame(({ clock }) => {
    if (!handTrackingEnabled || !isPresenting) return;
    
    // Process controller raycasting for far interaction
    if (interactionMode !== 'hands-only' && (leftControllerRef.current || rightControllerRef.current)) {
      // Perform raycasting from controllers to detect interactive objects
      if (leftControllerRef.current) {
        const origin = (raycasterRef.current.ray.origin as any);
        origin.setFromMatrixPosition((leftControllerRef.current as any).matrixWorld);
        
        const direction = (raycasterRef.current.ray.direction as any);
        direction.set(0, 0, -1);
        direction.applyMatrix4(
          new (THREE as any).Matrix4().extractRotation((leftControllerRef.current as any).matrixWorld)
        );
        
        const intersects = raycasterRef.current.intersectObjects(
          interactables.map(obj => obj.object), 
          false
        );
        
        if (intersects.length > 0) {
          handleInteractionEvent({
            type: 'hover',
            distance: 'far',
            controller: leftControllerRef.current as Object3D,
            hand: 'left',
            point: [
              (intersects[0].point as any).x,
              (intersects[0].point as any).y,
              (intersects[0].point as any).z
            ]
          });
        }
      }
      
      // Same for right controller
      if (rightControllerRef.current) {
        // Similar raycasting logic for right controller
      }
    }
    
    // Process hand tracking for near interaction
    if (interactionMode !== 'controllers-only' && (leftHandRef.current || rightHandRef.current)) {
      // Process joints and recognize gestures for left hand
      if (leftHandRef.current && (leftHandRef.current as any).children.length > 0) {
        const leftJoints = new Map<string, Object3D>();
        (leftHandRef.current as any).children.forEach((joint: any) => {
          leftJoints.set(joint.name, joint);
        });
        
        recognizeGestures('left', leftJoints);
      }
      
      // Process joints and recognize gestures for right hand
      if (rightHandRef.current && (rightHandRef.current as any).children.length > 0) {
        const rightJoints = new Map<string, Object3D>();
        (rightHandRef.current as any).children.forEach((joint: any) => {
          rightJoints.set(joint.name, joint);
        });
        
        recognizeGestures('right', rightJoints);
      }
    }
  });
  
  // Toggle hand visualization for debugging
  const toggleHandVisualization = () => {
    setVisualizeHands(!visualizeHands);
  };
  
  if (!handTrackingEnabled) return null;
  
  return (
    // Only the container group is rendered - the actual implementation is done in useFrame
    <group name="hand-interaction-system">
      {children}
    </group>
  );
};

// Hook for hand tracking in components
export const useHandTracking = () => {
  const { isPresenting } = useSafeXR();
  
  const [pinchState, setPinchState] = useState<{left: boolean, right: boolean}>({
    left: false,
    right: false
  });
  
  // Hand positions state
  const [handPositions, setHandPositions] = useState<{
    left: [number, number, number] | null,
    right: [number, number, number] | null
  }>({
    left: null,
    right: null
  });
  
  // Gesture state
  const [gestureState, setGestureState] = useState<GestureState>({
    left: { pinch: false, grip: false, point: false, thumbsUp: false },
    right: { pinch: false, grip: false, point: false, thumbsUp: false }
  });
  
  // Update hand positions and gestures state from the system
  useFrame(() => {
    // If we're not in XR mode, don't try to update anything
    if (!isPresenting) {
      return;
    }
    // This would be implemented to sync with the hand tracking system 
    // and update the hook's state based on the HandInteractionSystem
  });
  
  return {
    pinchState,
    handPositions,
    gestureState,
    isLeftHandVisible: !!handPositions.left,
    isRightHandVisible: !!handPositions.right
  };
};

// Interactable component that works with hand tracking
export const HandInteractable: React.FC<{
  children?: React.ReactNode,
  id?: string,
  onHover?: () => void,
  onUnhover?: () => void,
  onSelect?: () => void,
  position?: [number, number, number],
  scale?: [number, number, number]
}> = ({
  children,
  id,
  onHover,
  onUnhover,
  onSelect,
  position = [0, 0, 0],
  scale = [1, 1, 1]
}) => {
  const [isHovered, setIsHovered] = useState(false);
  
  const handlePointerOver = () => {
    setIsHovered(true);
    if (onHover) onHover();
  };
  
  const handlePointerOut = () => {
    setIsHovered(false);
    if (onUnhover) onUnhover();
  };
  
  const handleClick = () => {
    if (onSelect) onSelect();
  };
  
  return (
    <group
      position={position}
      name={id || 'interactable'}
      scale={scale}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    >
      {children}
      {isHovered && (
        // Create a simpler hover indicator without material props
        <group name="hover-indicator" scale={[1.05, 1.05, 1.05]}>
          {/* Using a primitive mesh for hover effects to avoid TypeScript errors */}
          {React.createElement('mesh', {
            children: [React.createElement('sphereGeometry', { args: [1, 16, 16] })]
          })}
        </group>
      )}
    </group>
  );
};

/**
 * This is a stub implementation for the Quest-specific XR features.
 * In a real implementation, this would integrate with the WebXR API
 * and Quest hand tracking capabilities.
 * 
 * Key features that would be implemented:
 * 1. Hand tracking using WebXR Hand Input API
 * 2. Gesture recognition (pinch, grip, point)
 * 3. Proper integration with react-three/fiber and react-three/xr
 * 4. AR passthrough mode specific to Quest devices
 * 5. Performance optimizations for XR
 */

// Import the safe XR hooks to prevent errors outside XR context
import { useSafeXR, withSafeXR } from '../hooks/useSafeXRHooks';

// Export the wrapped version as the default, which is safe to use anywhere
const SafeHandInteractionSystem = withSafeXR(HandInteractionSystem, 'HandInteractionSystem');

// Default export is now the safe version
export default SafeHandInteractionSystem;
