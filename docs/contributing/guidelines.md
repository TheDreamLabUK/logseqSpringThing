# Contributing Guidelines

Thank you for your interest in contributing to LogseqXR! This document provides guidelines and best practices for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](./CODE_OF_CONDUCT.md) to maintain a positive and inclusive community.

## Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/logseq-xr.git
cd logseq-xr
```

### 2. Set Up Development Environment
Follow the [Development Setup Guide](../development/setup.md) to configure your environment.

### 3. Create a Branch
```bash
# Create a branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-fix-name
```

## Development Workflow

### 1. Code Standards

#### Rust Code
- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting
- Run `clippy` for linting
```bash
# Format code
cargo fmt

# Run clippy
cargo clippy -- -D warnings
```

#### TypeScript/JavaScript Code
- Follow the project's ESLint configuration
- Use Prettier for formatting
```bash
# Lint code
pnpm lint

# Format code
pnpm format
```

### 2. Testing Requirements

#### Write Tests
- Add unit tests for new functionality
- Update existing tests when modifying code
- Ensure all tests pass before submitting PR

```bash
# Run Rust tests
cargo test

# Run TypeScript tests
pnpm test

# Run end-to-end tests
pnpm test:e2e
```

#### Test Coverage
- Aim for 80% or higher coverage
- Include both success and error cases
- Test edge cases and boundary conditions

### 3. Documentation

#### Code Documentation
- Document all public APIs
- Include examples in doc comments
- Update relevant documentation files

#### Rust Documentation
```rust
/// Calculates the force between two nodes
///
/// # Arguments
///
/// * `node1` - First node
/// * `node2` - Second node
///
/// # Returns
///
/// The force vector between the nodes
///
/// # Examples
///
/// ```
/// let force = calculate_force(&node1, &node2);
/// ```
pub fn calculate_force(node1: &Node, node2: &Node) -> Vector3<f32> {
    // Implementation
}
```

#### TypeScript Documentation
```typescript
/**
 * Updates node positions based on calculated forces
 * 
 * @param nodes - Array of nodes to update
 * @param forces - Calculated forces for each node
 * @param deltaTime - Time step for physics simulation
 * 
 * @returns Updated node positions
 * 
 * @example
 * ```typescript
 * const updatedNodes = updateNodePositions(nodes, forces, 0.016);
 * ```
 */
function updateNodePositions(
    nodes: Node[],
    forces: Vector3[],
    deltaTime: number
): Node[] {
    // Implementation
}
```

### 4. Commit Guidelines

#### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(graph): add force-directed layout algorithm

Implement Barnes-Hut algorithm for efficient force calculation
in large graphs. This improves performance for graphs with
over 1000 nodes.

Closes #123
```

### 5. Pull Request Process

1. **Update Your Branch**
```bash
git fetch origin
git rebase origin/main
```

2. **Check Your Changes**
```bash
# Run all checks
cargo fmt -- --check
cargo clippy
cargo test
pnpm lint
pnpm test
```

3. **Create Pull Request**
- Use the PR template
- Link related issues
- Provide clear description
- Include screenshots/videos if relevant

4. **PR Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added unit tests
- [ ] Updated existing tests
- [ ] Tested manually
- [ ] Added documentation

## Screenshots
(if applicable)

## Related Issues
Fixes #123
```

5. **Review Process**
- Address reviewer feedback
- Keep PR focused and manageable
- Maintain clear communication

## Best Practices

### 1. Code Organization
- Keep functions focused and small
- Use meaningful names
- Follow project structure
- Minimize dependencies

### 2. Performance
- Consider performance implications
- Profile code changes
- Document performance impacts
- Test with large datasets

### 3. Security
- Follow security best practices
- Validate inputs
- Handle errors appropriately
- Use safe API methods

### 4. Accessibility
- Follow WCAG guidelines
- Test with screen readers
- Provide keyboard navigation
- Support high contrast modes

## Getting Help

- Join our [Discord server](https://discord.gg/logseq-xr)
- Check existing issues and discussions
- Ask questions in PR comments
- Contact maintainers

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## Related Documentation
- [Development Setup](../development/setup.md)
- [Architecture Overview](../overview/architecture.md)
- [API Documentation](../api/rest.md)