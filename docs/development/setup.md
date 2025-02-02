# Development Setup Guide

This guide will help you set up your development environment for LogseqXR.

## Prerequisites

### Required Software
- Node.js v16+ (recommended: v20)
- Rust (version 1.82.0+)
- Docker & Docker Compose
- Git
- pnpm (recommended) or npm
- CUDA Toolkit 12.2+ (for GPU acceleration)

### Hardware Requirements
- Modern GPU with WebGPU support (recommended)
- At least 8GB RAM (16GB recommended)
- 4GB+ GPU memory for optimal performance

## Initial Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/logseq-xr.git
cd logseq-xr
```

### 2. Environment Configuration
Copy the environment template and configure your settings:
```bash
cp .env_template .env
```

Edit `.env` with your configuration:
```env
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repo_name
GITHUB_BASE_PATH=path/to/markdown/files
TUNNEL_TOKEN=your_cloudflare_tunnel_token
OPENWEATHER_API_KEY=your_api_key
```

### 3. Install Dependencies

#### Frontend Dependencies
```bash
# Install pnpm if not already installed
npm install -g pnpm@9.14.2

# Install frontend dependencies
pnpm install
```

#### Rust Dependencies
```bash
# Install Rust dependencies
cargo build
```

## Configuration Files

### settings.yaml
Create or modify `settings.yaml` in the project root:
```yaml
system:
  network:
    domain: localhost
    port: 4000
    bind_address: 0.0.0.0
  websocket:
    heartbeat_interval: 30
    reconnect_attempts: 5
  gpu:
    enable: true
    compute_shader: auto
    workgroup_size: 256

visualization:
  nodes:
    default_size: 1.0
    min_size: 0.5
    max_size: 2.0
  edges:
    default_width: 0.1
    min_width: 0.05
    max_width: 0.3
  physics:
    spring_strength: 0.1
    repulsion: 1.0
    damping: 0.8
```

### nginx.conf
The nginx configuration is provided in the repository. For local development, the default configuration should work without modifications.

## Development Workflow

### 1. Start Development Server
For local development without Docker:

```bash
# Terminal 1: Start the Rust backend
cargo watch -x run

# Terminal 2: Start the frontend development server
pnpm dev
```

### 2. Using Docker for Development
```bash
# Build and start all services
docker-compose up --build

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down
```

## Testing

### Running Tests

#### Frontend Tests
```bash
# Run all frontend tests
pnpm test

# Run with coverage
pnpm test:coverage

# Run end-to-end tests
pnpm test:e2e
```

#### Backend Tests
```bash
# Run all Rust tests
cargo test

# Run specific test
cargo test test_name

# Run with logging
RUST_LOG=debug cargo test
```

### Linting and Formatting

#### Frontend
```bash
# Run ESLint
pnpm lint

# Fix ESLint issues
pnpm lint:fix

# Format with Prettier
pnpm format
```

#### Backend
```bash
# Format Rust code
cargo fmt

# Run Clippy linter
cargo clippy
```

## Debugging

### Frontend Debugging
1. Open Chrome DevTools (F12)
2. Use the "Sources" tab to set breakpoints
3. Check the "Console" tab for errors and logs
4. Use the React DevTools extension for component debugging

### Backend Debugging
1. Set the `RUST_LOG` environment variable:
```bash
export RUST_LOG=debug
```

2. Use logging in your code:
```rust
debug!("Debug message");
info!("Info message");
error!("Error message");
```

3. Check logs in `/tmp/webxr.log`

### WebSocket Debugging
Use the browser's Network tab to inspect WebSocket messages:
1. Filter by "WS"
2. Click on the WebSocket connection
3. View messages in the "Messages" tab

## Common Issues

### GPU Initialization Failed
If GPU initialization fails, the system will fall back to CPU computation. Check:
1. GPU drivers are up to date
2. CUDA is properly installed
3. GPU has sufficient memory

### WebSocket Connection Issues
If WebSocket connection fails:
1. Check if the backend is running
2. Verify nginx configuration
3. Check browser console for errors
4. Ensure ports are not blocked by firewall

### GitHub API Rate Limiting
If you encounter GitHub API rate limiting:
1. Check your token permissions
2. Use a token with appropriate scopes
3. Implement request caching
4. Add rate limit handling

## Development Tools

### Recommended VSCode Extensions
- rust-analyzer
- ESLint
- Prettier
- Docker
- WebGL Shader
- Mermaid Preview
- Error Lens

### VSCode Settings
```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "rust-analyzer.checkOnSave.command": "clippy"
}
```

## Related Documentation
- [Technical Architecture](../overview/architecture.md)
- [API Documentation](../api/rest.md)
- [Contributing Guidelines](../contributing/guidelines.md)