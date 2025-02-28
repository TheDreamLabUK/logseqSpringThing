# Development Setup Guide

This guide will help you set up your development environment for LogseqXR.

## Prerequisites

### Required Software
- Node.js >=18.0.0
- Rust (version 1.70.0+)
- Docker & Docker Compose
- Git
- pnpm (recommended) or npm
- CUDA Toolkit 12.2+ (optional, for GPU acceleration)

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended)
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

Required environment variables:

#### Server Configuration
```env
RUST_LOG=info
BIND_ADDRESS=0.0.0.0
DEBUG_MODE=true
```

#### CUDA Configuration (Optional)
```env
CUDA_ARCH=86  # 89 for Ada/A6000
```

#### Network Configuration
```env
DOMAIN=your.domain.com
TUNNEL_TOKEN=your_cloudflare_tunnel_token
TUNNEL_ID=your_tunnel_id
```

#### GitHub Configuration
```env
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repo_name
GITHUB_PATH=/pages
GITHUB_VERSION=
GITHUB_RATE_LIMIT=
```

#### AI Integration Configuration
```env
# RAGFlow
RAGFLOW_API_KEY=your_api_key
RAGFLOW_API_BASE_URL=http://ragflow-server/v1/
RAGFLOW_TIMEOUT=30
RAGFLOW_MAX_RETRIES=3

# Perplexity
PERPLEXITY_API_KEY=your_api_key
PERPLEXITY_MODEL=llama-3.1-sonar-small-128k-online
PERPLEXITY_API_URL=https://api.perplexity.ai/chat/completions
PERPLEXITY_MAX_TOKENS=4096

# OpenAI
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=wss://api.openai.com/v1/realtime
```

#### Authentication Configuration
```env
# Base access control
APPROVED_PUBKEYS=pubkey1,pubkey2

# Role-based access
POWER_USER_PUBKEYS=pubkey1
SETTINGS_SYNC_ENABLED_PUBKEYS=pubkey1,pubkey2

# Feature-specific access
PERPLEXITY_ENABLED_PUBKEYS=pubkey1
OPENAI_ENABLED_PUBKEYS=pubkey1
RAGFLOW_ENABLED_PUBKEYS=pubkey1
```

### 3. Install Dependencies

#### Frontend Dependencies
```bash
# Install pnpm if not already installed
npm install -g pnpm

# Install frontend dependencies
pnpm install
```

#### Rust Dependencies
```bash
# Install Rust dependencies
cargo build

# For GPU support
cargo build --features gpu
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
    compression_enabled: true
    compression_threshold: 1024
  gpu:
    enable: true
    workgroup_size: 256
  debug:
    enabled: false
    enable_websocket_debug: false

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
# Run type checking
pnpm type-check

# Run linting
pnpm lint

# Format code
pnpm format
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
2. CUDA is properly installed (if using GPU features)
3. GPU has sufficient memory
4. Correct CUDA_ARCH is set in .env

### WebSocket Connection Issues
If WebSocket connection fails:
1. Check if the backend is running
2. Verify nginx configuration
3. Check browser console for errors
4. Ensure ports are not blocked by firewall
5. Verify WebSocket compression settings

### Authentication Issues
If authentication fails:
1. Verify Nostr public keys are correctly configured
2. Check role-based access settings
3. Verify feature-specific access permissions
4. Check WebSocket connection for authentication messages

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