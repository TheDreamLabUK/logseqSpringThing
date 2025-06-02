# Client-Side Visualization Concepts

This document outlines the higher-level concepts behind the visualization of the knowledge graph in the LogseqXR client. It explains *what* is being visualized and *how* different visual elements represent data, distinguishing itself from `rendering.md` which details the technical "how-to" of drawing these elements.

## Core Visualization Metaphor

The primary goal is to transform an abstract knowledge graph, typically represented by Markdown files and their links, into a tangible and interactive 3D spatial environment.

-   **Nodes as Entities:** Each primary piece of information (e.g., a Logseq page, a specific block, or a concept) is represented as a **node** in the 3D space.
    -   Typically, nodes correspond to individual Markdown files.
    -   The visual appearance of a node (size, color, shape) can be mapped to its attributes (e.g., file size, type, metadata tags).
-   **Edges as Relationships:** Links and connections between these entities (e.g., hyperlinks, block references, tags) are represented as **edges** connecting the corresponding nodes.
    -   The visual properties of edges (thickness, color, style) can signify the type or strength of the relationship.

## Mapping Data to Visual Elements

The effectiveness of the visualization hinges on how data attributes are mapped to visual properties.

### Node Visuals

-   **Size:**
    -   `visualisation.nodes.nodeSize` in [`settings.ts`](../../client/src/features/settings/config/settings.ts) provides a base size.
    -   This base size can be modulated by data attributes. For example, node size could represent:
        -   The word count or file size of a document.
        -   The number of connections (degree centrality).
        -   The importance score derived from an algorithm (e.g., PageRank).
-   **Color:**
    -   `visualisation.nodes.baseColor` provides a default color.
    -   Color can be used to categorize nodes, for instance:
        -   By file type (e.g., Markdown, PDF, image).
        -   By tags or keywords present in the metadata.
        -   By recency or modification date.
-   **Shape / Form:**
    -   While often spheres for simplicity and performance (especially with instancing), nodes can take different shapes to represent different types of entities.
    -   The `enableMetadataShape` setting in `NodeSettings` hints at the possibility of altering node geometry based on metadata.
-   **Holograms:**
    -   The `enableHologram` setting and `HologramSettings` in [`settings.ts`](../../client/src/features/settings/config/settings.ts) allow for a distinct visual style for certain nodes, potentially to highlight them or indicate a special status (e.g., recently accessed, important).
    -   The [`HologramManager.tsx`](../../client/src/features/visualisation/renderers/HologramManager.tsx) and [`HologramMaterial.tsx`](../../client/src/features/visualisation/renderers/materials/HologramMaterial.tsx) are responsible for this effect.

### Edge Visuals

-   **Thickness/Width:**
    -   `visualisation.edges.baseWidth` and `widthRange` can be used.
    -   Edge thickness can represent the strength or frequency of a connection.
-   **Color:**
    -   `visualisation.edges.color` provides a default.
    -   Can indicate the type of link (e.g., direct hyperlink vs. implicit relationship).
    -   Can use a gradient (`useGradient`, `gradientColors`) for aesthetic purposes or to show directionality.
-   **Style:**
    -   Arrows (`enableArrows`, `arrowSize`) can indicate directionality.
    -   Flow effects (`enableFlowEffect`, `flowSpeed`, `flowIntensity`) can visualize activity or information flow along connections.

### Text Labels

-   `LabelSettings` in [`settings.ts`](../../client/src/features/settings/config/settings.ts) control the appearance of text.
-   Node labels typically display the title or filename.
-   Edge labels (if enabled) could show the type of relationship.

## Interactive Visualization

The visualization is not static; users can interact with it:

-   **Navigation:** Panning, zooming, and rotating the camera to explore the graph from different perspectives.
-   **Selection:** Clicking on nodes or edges to highlight them and display more detailed information (e.g., in a side panel).
-   **Filtering/Searching:** Dynamically showing/hiding nodes and edges based on criteria.
-   **Spatial Arrangement:** The layout of nodes in 3D space is crucial. This is typically handled by a server-side physics simulation (`GraphService` in Rust) that attempts to position connected nodes closer together and push unrelated nodes apart, revealing clusters and structures.

## Metadata Visualization

### `MetadataVisualizer.tsx` (`client/src/features/visualisation/components/MetadataVisualizer.tsx`)

This component is responsible for displaying additional information or visual cues based on the metadata associated with nodes.

**Possible Implementations:**
-   **Icons/Glyphs:** Displaying small icons on or near nodes to represent file types, tags, or status.
-   **Auras/Halos:** Using subtle visual effects around nodes to indicate certain metadata properties (e.g., a glowing aura for unread items).
-   **Dynamic Text Panels:** Showing detailed metadata in a 2D overlay when a node is selected or hovered.

## Distinction from `rendering.md`

-   **`visualization.md` (this document):** Focuses on the *conceptual* aspects.
    -   What do nodes and edges *represent*?
    -   How is data (size, type, connections) *encoded* into visual properties (size, color, shape)?
    -   What insights is the user intended to gain from these visual mappings?
    -   The *meaning* behind the visual design choices.
-   **[`rendering.md`](./rendering.md):** Focuses on the *technical implementation*.
    -   How are spheres, lines, and text *drawn* using React Three Fiber and Three.js?
    -   What specific components (`GraphCanvas`, `GraphManager`, `TextRenderer`) are involved?
    -   What techniques (instancing, shaders like `HologramMaterial.tsx`) are used for performance and visual effects?
    -   The *mechanics* of putting pixels on the screen.

In essence, `visualization.md` is about the "language" of the visual representation, while `rendering.md` is about the "grammar and tools" used to speak that language.