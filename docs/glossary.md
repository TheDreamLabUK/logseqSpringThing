# Glossary

This glossary defines key terms used throughout the LogseqSpringThing documentation to ensure consistency.

## A

**Actor** - In the context of this project, refers to Actix actors used for concurrent, message-based processing in the Rust backend.

**AppFullSettings** - The complete application settings structure containing all configuration options for both server and client components.

**Audio Processor** - Server-side component that handles audio data processing, including base64 decoding and JSON response parsing from AI services.

## B

**Binary Protocol** - The efficient 28-byte per node wire format used for real-time position updates over WebSocket connections.

**BinaryNodeData** - Server-side data structure containing node position, velocity, and physics properties (mass, flags).

## C

**Client Manager** - Server component that tracks connected WebSocket clients and handles message broadcasting.

**Command Palette** - Keyboard-driven interface for quickly accessing application features (similar to VS Code's command palette).

**Compression Threshold** - Minimum message size (default 1KB) before zlib compression is applied to WebSocket messages.

## E

**Edge Data** - Data structure representing connections between nodes in the graph, including source, target, and weight.

## G

**GPU Compute** - CUDA-based acceleration for graph physics calculations, managed by the GPUCompute struct.

**Graph Data Manager** - Client-side service responsible for managing graph state and synchronizing with the server.

## H

**Heartbeat Interval** - WebSocket ping/pong interval (30 seconds) to maintain connection and detect disconnections.

**Help System** - Context-sensitive help infrastructure providing tooltips, documentation, and interactive guides.

## K

**Kokoro** - Text-to-Speech (TTS) service provider integrated into the voice system.

## N

**Node** - Basic unit in the knowledge graph, representing a concept, idea, or piece of information.

**Nostr** - Decentralized authentication protocol used for user authentication and session management.

## O

**Onboarding Flow** - Multi-step guided tours helping new users understand application features.

## P

**Power User** - Authenticated user with elevated privileges, able to modify global server settings.

**Protected Settings** - Sensitive configuration data (API keys, user profiles) stored securely on the server.

## R

**RAGFlow** - Retrieval-Augmented Generation service for AI-powered question answering about the knowledge graph.

## S

**Settings Store** - Client-side Zustand store managing application settings and preferences.

**Simulation Parameters** - Configuration values controlling the physics simulation (spring strength, damping, etc.).

**Socket Flow** - The WebSocket message flow system handling real-time communication between client and server.

**Speech Service** - Unified service managing both Text-to-Speech (TTS) and Speech-to-Text (STT) operations.

## T

**Three.js** - 3D graphics library used for rendering the knowledge graph visualization.

**Todo System** - Task tracking system used during development for managing implementation steps.

## U

**UISettings** - Subset of settings specifically related to user interface preferences and visualization options.

## V

**Vec3Data** - Three-dimensional vector type used for positions and velocities, compatible with both CUDA and WebSocket protocols.

**Voice System** - Integrated voice interaction system supporting both speech input and output.

## W

**WebSocket Service** - Client-side singleton service managing WebSocket connections and message handling.

**WebXR** - Web-based Virtual Reality (VR) and Augmented Reality (AR) API for immersive experiences.

**Whisper** - Speech-to-Text (STT) service provider integrated into the voice system.

**Wire Format** - The exact binary layout of data transmitted over WebSocket connections.

## X

**XR Controller** - Component managing WebXR sessions and immersive mode transitions.

## Z

**Zustand** - Lightweight state management library used for client-side state management.