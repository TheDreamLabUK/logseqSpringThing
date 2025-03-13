# Graph System Architecture

## Overview
The graph system manages the application's core data structure, providing thread-safe access and real-time updates.

## Components

### Graph Data Structure
```rust
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: HashMap<String, String>,
}
```

### Node Management
```rust
pub struct Node {
    pub id: String,
    pub data: NodeData,
    pub metadata: NodeMetadata,
}
```

### Edge Management
```rust
pub struct Edge {
    pub source: String,
    pub target: String,
    pub weight: f32,
    pub metadata: EdgeMetadata,
}
```

## Services Integration

### Graph Service
```rust
pub struct GraphService {
    settings: Arc<RwLock<Settings>>,
    graph_data: Arc<RwLock<GraphData>>,
    node_map: Arc<RwLock<HashMap<String, Node>>>,
    gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
}
```

### Metadata Integration
```rust
pub async fn build_graph_from_metadata(
    metadata_store: &MetadataStore
) -> Result<GraphData, Error>
```

## Real-time Updates

### WebSocket Integration
- Live graph updates
- State synchronization
- Client notifications

### Concurrency Management
```rust
static GRAPH_REBUILD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static SIMULATION_LOOP_RUNNING: AtomicBool = AtomicBool::new(false);
```

## Performance Optimization

### Batch Processing
```rust
const BATCH_SIZE: usize = 5;
for chunk in github_files.chunks(BATCH_SIZE) {
    // Process files in batches
}
```

### Memory Management
- Efficient state updates
- Resource cleanup
- Cache optimization

## Error Handling

### Graph Operations
- Node/edge validation
- Data consistency checks
- Error recovery

### State Management
```rust
pub async fn calculate_layout_with_retry(
    gpu_compute: &Arc<RwLock<GPUCompute>>,
    graph: &mut GraphData,
    node_map: &mut HashMap<String, Node>, 
    params: &SimulationParams,
) -> std::io::Result<()>
```

## Monitoring

### Performance Metrics
- Graph size tracking
- Update frequency
- Resource utilization

### Health Checks
- Data consistency
- Service availability
- Resource status