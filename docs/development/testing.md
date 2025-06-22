# Testing Documentation

## Overview

LogseqSpringThing uses a comprehensive testing strategy covering unit tests, integration tests, and end-to-end tests across both Rust server and TypeScript client codebases.

## Test Structure

```
├── src/
│   └── utils/
│       └── tests/
│           └── socket_flow_tests.rs
├── src/config/
│   └── feature_access_test.rs
├── client/
│   └── src/
│       └── __tests__/
└── scripts/
    └── test.sh
```

## Server Testing (Rust)

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_feature_access

# Run tests in watch mode
cargo watch -x test
```

### Unit Tests

Unit tests are located alongside the code they test using Rust's built-in test framework.

#### Example: Feature Access Tests

**Location**: `src/config/feature_access_test.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_enabled() {
        let config = FeatureAccessConfig::default();
        assert!(config.is_feature_enabled("graph_visualization"));
    }

    #[test]
    fn test_power_user_access() {
        let mut config = FeatureAccessConfig::default();
        config.add_power_user("user123");
        assert!(config.is_power_user("user123"));
    }
}
```

#### Example: Binary Protocol Tests

**Location**: `src/utils/tests/socket_flow_tests.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_encoding() {
        let node_data = BinaryNodeData {
            position: Vec3Data { x: 1.0, y: 2.0, z: 3.0 },
            velocity: Vec3Data { x: 0.1, y: 0.2, z: 0.3 },
        };
        
        let encoded = encode_binary_data(&node_data);
        let decoded = decode_binary_data(&encoded).unwrap();
        
        assert_eq!(node_data, decoded);
    }
}
```

### Integration Tests

Integration tests for actor systems and services.

```rust
#[actix_rt::test]
async fn test_graph_actor_integration() {
    // Initialize actors
    let client_manager = ClientManagerActor::new().start();
    let graph_actor = GraphServiceActor::new(client_manager, None).start();
    
    // Test actor messages
    let result = graph_actor.send(GetGraphData).await.unwrap();
    assert!(result.is_ok());
    
    // Test node operations
    let node = Node::new(1, "Test".to_string());
    let add_result = graph_actor.send(AddNode { node }).await.unwrap();
    assert!(add_result.is_ok());
}
```

### API Integration Tests

```rust
#[actix_web::test]
async fn test_api_endpoints() {
    let app = test::init_service(
        App::new()
            .app_data(create_test_state())
            .configure(configure_routes)
    ).await;

    let req = test::TestRequest::get()
        .uri("/api/graph")
        .to_request();
        
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}
```

## Client Testing (TypeScript)

### Setup

```json
// package.json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage"
  },
  "devDependencies": {
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "vitest": "^1.0.0",
    "@vitest/ui": "^1.0.0"
  }
}
```

### Running Tests

```bash
# Run all tests
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run in watch mode
npm test -- --watch
```

### Component Tests

```typescript
// client/src/__tests__/components/GraphCanvas.test.tsx
import { render, screen } from '@testing-library/react';
import { GraphCanvas } from '@/features/graph/components/GraphCanvas';

describe('GraphCanvas', () => {
  it('renders without crashing', () => {
    render(<GraphCanvas />);
    expect(screen.getByTestId('graph-canvas')).toBeInTheDocument();
  });

  it('initializes WebGL context', () => {
    const { container } = render(<GraphCanvas />);
    const canvas = container.querySelector('canvas');
    expect(canvas).toHaveAttribute('webgl');
  });
});
```

### Hook Tests

```typescript
// client/src/__tests__/hooks/useAuth.test.ts
import { renderHook, act } from '@testing-library/react';
import { useAuth } from '@/features/auth/hooks/useAuth';

describe('useAuth', () => {
  it('handles login flow', async () => {
    const { result } = renderHook(() => useAuth());
    
    await act(async () => {
      await result.current.login();
    });
    
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toBeDefined();
  });
});
```

### Service Tests

```typescript
// client/src/__tests__/services/WebSocketService.test.ts
import { WebSocketService } from '@/services/WebSocketService';

describe('WebSocketService', () => {
  let service: WebSocketService;
  
  beforeEach(() => {
    service = new WebSocketService();
  });
  
  it('connects to server', async () => {
    const mockServer = new WS('ws://localhost:8080');
    
    await service.connect();
    await mockServer.connected;
    
    expect(service.isConnected).toBe(true);
  });
});
```

## Testing Strategies

### Actor System Testing

1. **Message Testing**: Verify all actor messages are handled correctly
2. **State Testing**: Ensure actor state remains consistent
3. **Concurrency Testing**: Test concurrent message handling
4. **Failure Testing**: Verify graceful error handling

### WebSocket Testing

```rust
#[tokio::test]
async fn test_websocket_binary_protocol() {
    let server = create_test_server().await;
    let client = connect_test_client(&server).await;
    
    // Send binary message
    let node_data = create_test_node_data();
    let binary_msg = encode_node_update(node_data);
    client.send(Message::Binary(binary_msg)).await.unwrap();
    
    // Verify broadcast
    let received = client.recv().await.unwrap();
    assert!(matches!(received, Message::Binary(_)));
}
```

### GPU Testing

```rust
#[tokio::test]
async fn test_gpu_fallback() {
    // Force GPU failure
    std::env::set_var("CUDA_VISIBLE_DEVICES", "");
    
    let gpu_compute = GPUCompute::new(test_graph()).await;
    assert!(gpu_compute.is_err() || gpu_compute.unwrap().cpu_fallback_active);
}
```

## Mocking and Test Utilities

### Mock Services

```rust
// src/tests/mocks/mod.rs
pub fn create_mock_ragflow_service() -> RAGFlowService {
    RAGFlowService::new_with_client(
        create_mock_http_client(),
        test_settings()
    )
}

pub fn create_mock_http_client() -> Client {
    // Return client with mock responses
}
```

### Test Fixtures

```rust
// src/tests/fixtures/mod.rs
pub fn create_test_graph() -> GraphData {
    GraphData {
        nodes: vec![
            Node::new(1, "Node 1"),
            Node::new(2, "Node 2"),
        ],
        edges: vec![
            Edge::new("edge1", 1, 2),
        ],
    }
}
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      - name: Run tests
        run: cargo test --all-features

  client-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: cd client && npm ci
      - run: cd client && npm test
```

## Test Coverage

### Rust Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage
```

### TypeScript Coverage

```bash
# Run with coverage
npm run test:coverage

# View report
open coverage/index.html
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
   ```rust
   #[test]
   fn test_graph_actor_handles_concurrent_updates() { }
   ```

2. **Test Organization**: Group related tests in modules
   ```rust
   mod graph_tests {
       #[test]
       fn test_add_node() { }
       
       #[test]
       fn test_remove_node() { }
   }
   ```

3. **Isolation**: Each test should be independent
   ```rust
   #[test]
   fn test_isolated_behavior() {
       let state = create_fresh_state();
       // Test logic
   }
   ```

4. **Async Testing**: Use appropriate async test macros
   ```rust
   #[tokio::test]
   async fn test_async_operation() {
       let result = async_function().await;
       assert!(result.is_ok());
   }
   ```

5. **Error Cases**: Always test error scenarios
   ```rust
   #[test]
   fn test_invalid_input_returns_error() {
       let result = process_invalid_input();
       assert!(matches!(result, Err(Error::InvalidInput)));
   }
   ```

## Debugging Tests

### Rust Test Debugging

```bash
# Run single test with output
RUST_LOG=debug cargo test test_name -- --nocapture

# Use debugger
rust-gdb target/debug/test_binary
```

### TypeScript Test Debugging

```typescript
// Add debugger statement
it('debug test', () => {
  debugger; // Breakpoint here
  expect(true).toBe(true);
});
```

## Performance Testing

### Load Testing

```rust
#[test]
fn test_graph_performance() {
    let mut graph = GraphData::new();
    
    // Add many nodes
    let start = std::time::Instant::now();
    for i in 0..10000 {
        graph.add_node(Node::new(i, format!("Node {}", i)));
    }
    let duration = start.elapsed();
    
    assert!(duration.as_millis() < 100); // Should complete in 100ms
}
```

### Benchmarking

```rust
#[bench]
fn bench_force_calculation(b: &mut Bencher) {
    let graph = create_large_test_graph();
    b.iter(|| {
        compute_forces(&graph)
    });
}
```

## Related Documentation

- [Development Setup](./setup.md) - Environment configuration
- [Debugging](./debugging.md) - Debugging techniques
- [CI/CD](../deployment/index.md) - Continuous integration setup