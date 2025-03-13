# Performance Optimizations

This document details performance optimizations implemented in the application to improve startup time, reduce memory usage, and enhance UI responsiveness.

## Server-Side Optimizations

### Metadata Caching

#### Individual File Metadata
- **Description**: Store metadata for each file separately
- **Implementation**: Files are stored at `/app/data/metadata/files/<filename>.json`
- **Benefit**: Only files that have changed need to be reprocessed, rather than rebuilding the entire metadata store when any file changes
- **Validation**: SHA1 hash-based validation ensures we only reprocess files that have actually changed

#### Graph Data Caching
- **Description**: Cache the entire graph structure (nodes and edges)
- **Implementation**: Serialized to disk at `/app/data/metadata/graph.json`
- **Benefit**: Avoid rebuilding the entire graph on startup when metadata hasn't changed
- **Performance Impact**: Startup is significantly faster for subsequent runs

#### Layout Position Caching
- **Description**: Preserve the calculated node positions between sessions
- **Implementation**: Stored at `/app/data/metadata/layout.json`
- **Benefit**: Preserves user's mental map of the graph between sessions
- **Details**: Includes x,y,z coordinates for each node indexed by node ID

### Lazy Initialization

- **Description**: Defer expensive operations until they're actually needed
- **Implementation**: 
  - Graph is only built when first requested by a client
  - Layout calculation is no longer performed during server startup
  - Initial 500ms startup delay was removed
- **Benefit**: Server starts much faster and is immediately responsive
- **Details**: Uses caches where available, falls back to full calculation only when necessary

### Temporarily Disabled Services

- **Perplexity Service**: Temporarily commented out as it's not currently in use
- **Benefits**: Reduces API calls and response times

## Client-Side Considerations

- Clients should be prepared for lazily initialized data
- First request may take longer as the graph is built
- Subsequent requests will be faster as they use cached values

## Future Improvements

- **Background Updates**: Implement a background task to periodically check for changes
- **Incremental Updates**: Support partial graph updates when only a few files change
- **Differential Response**: Send only changed data to clients
- **Memory Management**: Add cache size limits and cleanup of old cached data

## Metrics

Initial measurements show these optimizations provide:
- Reduced server startup time: From ~5s to <1s
- Reduced memory usage during startup
- Improved responsiveness for WebSocket communications
- Consistent graph layouts for better user experience