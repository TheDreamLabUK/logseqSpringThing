# Client Core Utilities and Hooks

This document describes the core utility functions and general-purpose React hooks used across the LogseqXR client application. These modules provide foundational functionalities that support various components and features.

## Core Utilities (`client/src/utils/`)

The `client/src/utils/` directory contains a collection of general-purpose utility functions designed for reusability and to encapsulate common logic.

### `logger.ts`

Provides a centralized logging mechanism for the client application. It allows for structured logging with different severity levels (e.g., `debug`, `info`, `warn`, `error`) and can be configured to control log output based on application settings.

**Key Features:**
-   **Level-based logging:** Filter messages based on their importance.
-   **Contextual logging:** Include additional data with log messages for better debugging.
-   **Conditional logging:** Enable/disable logging based on environment or debug flags.

**Usage Example:**
```typescript
import { logger } from './logger';

logger.info('Application started successfully.');
logger.debug('Processing data:', { data: someObject });
logger.error('Failed to fetch graph data:', error);
```

### `binaryUtils.ts`

Contains helper functions for working with binary data, particularly useful for encoding and decoding messages transmitted over WebSockets using the custom binary protocol.

**Key Features:**
-   Functions for converting between `ArrayBuffer`, `Float32Array`, and other binary representations.
-   Utilities for reading and writing specific data types (e.g., floats, integers) from/to `ArrayBuffer` views.

### `caseConversion.ts`

Provides utilities for converting string cases, such as `camelCase` to `snake_case` and vice-versa. This is often used when interacting with API endpoints or data structures that follow different naming conventions (e.g., JavaScript frontend using `camelCase` and Rust backend using `snake_case`).

**Key Functions:**
-   `camelToSnakeCase(str: string): string`
-   `snakeToCamelCase(str: string): string`

### `cn.ts`

A small utility for conditionally joining CSS class names, often used with Tailwind CSS to build dynamic class strings. It's a common pattern for managing component styling based on props or state.

**Usage Example:**
```typescript
import { cn } from './cn';

const isActive = true;
const buttonClasses = cn('btn', isActive && 'btn-active', 'px-4');
// buttonClasses might be "btn btn-active px-4"
```

### `debugState.ts`

Provides utilities for managing and inspecting the application's debug state. This can include flags for enabling/disabling various debug visualizations, logging verbosity, or performance overlays.

**Key Features:**
-   Functions to set and retrieve debug flags.
-   Integration with the `SettingsStore` for persistent debug settings.

### `deepMerge.ts`

A utility function for performing a deep merge of two or more objects. This is useful for combining default settings with user-specific overrides or for merging complex configuration objects.

**Key Features:**
-   Recursively merges properties of source objects into a target object.
-   Handles various data types, including arrays and nested objects.

### `objectPath.ts`

Provides functions for safely accessing, setting, or deleting values within nested JavaScript objects using a dot-separated path string (e.g., `'visualisation.nodes.size'`). This is particularly useful for interacting with the `SettingsStore` where settings are often accessed by path.

**Key Functions:**
-   `get(obj: object, path: string, defaultValue?: any): any`
-   `set(obj: object, path: string, value: any): object`
-   `del(obj: object, path: string): object`

### `utils.ts`

A general-purpose utility file that might contain miscellaneous helper functions that don't fit into more specific categories. This could include array manipulation, string formatting, or other common operations.

## General-Purpose Hooks (`client/src/hooks/`)

The `client/src/hooks/` directory contains custom React hooks that provide reusable logic for common UI patterns or data management.

### `useContainerSize.ts`

A custom React hook that measures and provides the current dimensions (width and height) of a DOM element. It typically uses `ResizeObserver` to react to changes in the element's size, making it useful for responsive layouts or canvas sizing.

**Usage Example:**
```typescript
import React, { useRef } from 'react';
import { useContainerSize } from './useContainerSize';

function MyComponent() {
  const ref = useRef(null);
  const { width, height } = useContainerSize(ref);

  return (
    <div ref={ref} style={{ width: '100%', height: '500px' }}>
      Container size: {width}x{height}
    </div>
  );
}
```

### `useWindowSize.ts`

A custom React hook that provides the current dimensions (width and height) of the browser window. It updates reactively when the window is resized, enabling responsive design and layout adjustments based on the viewport size.

**Usage Example:**
```typescript
import { useWindowSize } from './useWindowSize';

function ResponsiveHeader() {
  const { width, height } = useWindowSize();
  const isMobile = width < 768;

  return (
    <header>
      {isMobile ? 'Mobile Header' : 'Desktop Header'}
      <p>Window size: {width}x{height}</p>
    </header>
  );
}