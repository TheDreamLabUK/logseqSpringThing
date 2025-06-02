# WebXR Integration

This document details the WebXR integration within the LogseqXR client, enabling immersive virtual and augmented reality experiences. It covers the setup, key components, interaction handling, and how settings affect XR mode.

## Overview

The WebXR integration allows users to explore their knowledge graph in a fully immersive 3D environment using VR headsets or AR-compatible devices. It leverages the `@react-three/xr` library, which provides React components and hooks for building WebXR experiences with React Three Fiber.

## Core XR Setup and Components

The XR functionality is primarily managed within the `client/src/features/xr/` directory.

### `@react-three/xr`

This library is fundamental to the XR integration. It provides:
-   `<XR />` component: The main wrapper to enable WebXR capabilities in an R3F scene.
-   `<Controllers />`, `<Hands />`: Components for rendering and interacting with VR controllers and tracked hands.
-   Hooks like `useXR()`, `useController()`, `useHand()`: For accessing XR session state, controller input, and hand tracking data.

### `XRController.tsx` ([`client/src/features/xr/components/XRController.tsx`](../../client/src/features/xr/components/XRController.tsx))

This component acts as the central orchestrator for the XR experience.

**Responsibilities:**
-   Wraps the main 3D scene with the `<XR />` component from `@react-three/xr`.
-   Manages the conditional rendering of XR-specific UI and interaction elements.
-   Integrates other XR components like [`XRScene.tsx`](../../client/src/features/xr/components/XRScene.tsx) and [`HandInteractionSystem.tsx`](../../client/src/features/xr/systems/HandInteractionSystem.tsx).
-   Handles XR session initialization and teardown logic via [`xrInitializer.ts`](../../client/src/features/xr/managers/xrInitializer.ts) and [`xrSessionManager.ts`](../../client/src/features/xr/managers/xrSessionManager.ts).

### `XRScene.tsx` ([`client/src/features/xr/components/XRScene.tsx`](../../client/src/features/xr/components/XRScene.tsx))

This component contains the 3D content that will be rendered in XR. It often reuses or adapts components from the main 2D desktop visualization.

**Responsibilities:**
-   Renders the graph (nodes, edges) within the XR environment.
-   May include XR-specific UI elements (e.g., floating menus, gaze cursors).
-   Adapts lighting and camera setup for optimal viewing in VR/AR.

### `XRVisualisationConnector.tsx` ([`client/src/features/xr/components/XRVisualisationConnector.tsx`](../../client/src/features/xr/components/XRVisualisationConnector.tsx))

This component acts as a bridge, connecting the core graph visualization logic (e.g., `GraphManager.tsx`) to the XR environment.

**Responsibilities:**
-   Passes graph data and updates to components rendered within `XRScene.tsx`.
-   May transform or adapt data for XR-specific presentation.
-   Ensures that interactions in XR mode correctly affect the underlying graph data.

### `HandInteractionSystem.tsx` ([`client/src/features/xr/systems/HandInteractionSystem.tsx`](../../client/src/features/xr/systems/HandInteractionSystem.tsx))

Manages interactions using tracked hands in WebXR.

**Responsibilities:**
-   Uses `@react-three/xr`'s `<Hands />` component and `useHand()` hook.
-   Detects hand gestures (e.g., pinch, grab) for object manipulation or UI interaction.
-   Provides visual feedback for hand tracking (e.g., rendering hand models or pointers).
-   Translates hand movements and gestures into actions within the application (e.g., selecting nodes, navigating menus).

### `xrInitializer.ts` ([`client/src/features/xr/managers/xrInitializer.ts`](../../client/src/features/xr/managers/xrInitializer.ts))

Contains logic for initializing the WebXR session.

**Responsibilities:**
-   Checks for WebXR browser support.
-   Requests an XR session (e.g., `immersive-vr` or `immersive-ar`).
-   Handles session feature requests (e.g., hand tracking, plane detection).

### `xrSessionManager.ts` ([`client/src/features/xr/managers/xrSessionManager.ts`](../../client/src/features/xr/managers/xrSessionManager.ts))

Manages the lifecycle of an active WebXR session.

**Responsibilities:**
-   Handles session start, end, and visibility change events.
-   Provides access to the current `XRSession` object.
-   Coordinates updates based on session state.

### `SafeXRProvider.tsx` ([`client/src/features/xr/providers/SafeXRProvider.tsx`](../../client/src/features/xr/providers/SafeXRProvider.tsx)) and `XRContextWrapper.tsx` ([`client/src/features/xr/providers/XRContextWrapper.tsx`](../../client/src/features/xr/providers/XRContextWrapper.tsx))

These components provide a React Context for sharing XR-related state and utilities throughout the XR part of the application.

**Responsibilities:**
-   `SafeXRProvider`: Checks for WebXR availability and gracefully handles cases where WebXR is not supported, preventing errors.
-   `XRContextWrapper`: Provides context values like the current XR session, controller states, or hand tracking data to descendant components.

## XR Interaction Handling

Interactions in XR are typically handled through:

-   **VR Controllers:** Using `@react-three/xr`'s `<Controllers />` and `useController()` hook to detect button presses, trigger pulls, and joystick movements. These inputs are then mapped to actions like selection, teleportation, or menu navigation.
-   **Hand Tracking:** Using `<Hands />` and `useHand()` along with `HandInteractionSystem.tsx` to interpret hand gestures (e.g., pinch-to-select, grab-to-move) and hand positions for direct manipulation of objects or UI elements.
-   **Gaze/Focus:** For simpler interactions or as a fallback, gaze-based selection (where the user looks at an object for a short duration to select it) can be implemented.

## Settings Affecting XR Mode

Several settings from the application's `SettingsStore` can influence the XR experience. These are found under the `xr` key in the [`Settings`](../../client/src/features/settings/config/settings.ts) interface, defined in [`client/src/features/settings/config/settings.ts`](../../client/src/features/settings/config/settings.ts).

-   **`xr.enabled`**: A global toggle for enabling/disabling XR mode (often a server-side setting that the client respects).
-   **`xr.clientSideEnableXR`**: A key client-side toggle that allows the user to enable/disable attempting to enter XR mode, independent of the server's `xr.enabled` setting. This is useful for client devices that may not support XR well.
-   **`xr.mode`**: Specifies the desired XR mode (e.g., `'inline'`, `'immersive-vr'`, `'immersive-ar'`).
-   **`xr.quality`**: Adjusts rendering quality for performance in XR.
-   **`xr.enableHandTracking`**: Toggles the use of hand tracking (Note: the setting name is `enableHandTracking`, not `handTracking`).
-   **`xr.handMeshEnabled`**, **`xr.handMeshColor`**: Controls the visual representation of tracked hands.
-   **Locomotion settings** (e.g., `xr.locomotionMethod`, `xr.movementSpeed`): Configure how users move within the virtual environment.
-   **AR-specific settings** (e.g., `xr.enablePlaneDetection`, `xr.passthroughOpacity`): For augmented reality experiences.

These settings are typically accessed within XR components via the `useSettingsStore` hook and applied to configure the `@react-three/xr` components or custom interaction logic.