# WebSocket Binary Protocol

This document describes the binary protocol used for efficient real-time updates of node positions and velocities over WebSockets.

## Overview

The binary protocol is designed to minimize bandwidth usage while providing fast updates for node positions and velocities in the 3D visualization. The protocol uses a fixed-size format for each node to simplify parsing and ensure consistency.

## Protocol Format

Each binary message consists of a series of node updates, where each node update is exactly 28 bytes:

| Field    | Type      | Size (bytes) | Description                       |
|----------|-----------|--------------|-----------------------------------|
| Node ID  | uint32    | 4            | Unique identifier for the node    |
| Position | float32[3]| 12           | X, Y, Z coordinates               |
| Velocity | float32[3]| 12           | X, Y, Z velocity components       |

Total: 28 bytes per node

## Compression

For large updates (more than 1KB), the binary data is compressed using zlib compression. The client automatically detects and decompresses these messages using the pako library.

## Server-Side Only Fields

The server maintains additional data for each node that is not transmitted over the wire:

- `mass` (u8): Node mass used for physics calculations
- `flags` (u8): Bit flags for node properties
- `padding` (u8[2]): Reserved for future use

These fields are used for server-side physics calculations and GPU processing but are not transmitted to clients to optimize bandwidth.

## Flow Sequence

1. Client connects to WebSocket endpoint (`/wss`)
2. Server sends a text message: `{"type": "connection_established"}`
3. Client sends a text message: `{"type": "requestInitialData"}`
4. Server starts sending binary updates at regular intervals (configured by `binary_update_rate` setting)
5. Server sends a text message: `{"type": "updatesStarted"}`
6. Client processes binary updates and updates the visualization

## Error Handling

If a binary message has an invalid size (not a multiple of 28 bytes), the client will log an error and discard the message. The server includes additional logging to help diagnose issues with binary message transmission.

## Implementation Notes

- All numeric values use little-endian byte order
- Position and velocity values are clamped to reasonable ranges on the client side
- The client validates all incoming data to prevent invalid values from affecting the visualization

## Debugging

To enable WebSocket debugging:

1. Set `system.debug.enabled = true` in settings.toml
2. Set `system.debug.enable_websocket_debug = true` in settings.toml

This will enable detailed logging of WebSocket messages on both client and server.