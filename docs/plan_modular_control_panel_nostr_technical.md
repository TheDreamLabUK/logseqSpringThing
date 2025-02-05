# Modular Control Panel with Nostr Authentication

## Implementation Status

### Completed Components

1. **Core Settings Architecture**
   - Base settings types and interfaces
   - Settings validation system
   - Settings store with local persistence
   - Real-time settings observer
   - Settings event emitter for cross-component communication

2. **UI Components**
   - Modular control panel with detachable sections
   - Real-time preview system
   - Advanced/Basic settings categorization
   - Drag-and-drop section management
   - Layout persistence

3. **Visualization Integration**
   - Real-time 3D visualization updates
   - Settings preview system
   - Performance-optimized update batching

### Authentication and Settings Inheritance

#### Unauthenticated Users
- Use browser's localStorage for settings persistence
- Settings are stored locally and not synced
- Default to basic settings visibility
- Limited to local visualization features
- Settings format:
```typescript
interface LocalSettings {
    settings: Settings;
    timestamp: number;
    version: string;
}
```

#### Authenticated Users (Nostr)
- Inherit settings from server's settings.yaml
- Settings are synced across all authenticated users
- Access to advanced settings based on role
- Settings format:
```typescript
interface ServerSettings {
    settings: Settings;
    timestamp: number;
    version: string;
    pubkey: string;
    role: 'user' | 'power_user';
}
```

#### Power Users
- Full access to all settings
- Can modify server's settings.yaml
- Access to advanced API features:
  - Perplexity API for AI assistance
  - RagFlow for document processing
  - GitHub integration for PR management
  - OpenAI voice synthesis
- Settings modifications are persisted to settings.yaml

## Settings Inheritance Flow

1. **Initial Load**
   ```mermaid
   graph TD
       A[Start] --> B{Authenticated?}
       B -->|No| C[Load Local Settings]
       B -->|Yes| D[Load Server Settings]
       D --> E{Is Power User?}
       E -->|No| F[Apply Read-Only]
       E -->|Yes| G[Enable Full Access]
   ```

2. **Settings Sync**
   ```mermaid
   graph TD
       A[Setting Changed] --> B{Authenticated?}
       B -->|No| C[Save Locally]
       B -->|Yes| D{Is Power User?}
       D -->|No| E[Preview Only]
       D -->|Yes| F[Update Server]
       F --> G[Sync to All Users]
   ```

## Remaining Tasks

1. **Authentication Integration**
   - [ ] Implement Nostr login component
   - [ ] Add role-based access control
   - [ ] Set up settings sync middleware

2. **Server-Side Implementation**
   - [ ] Create settings sync endpoint
   - [ ] Implement settings.yaml persistence
   - [ ] Add validation for power user operations

3. **API Feature Integration**
   - [ ] Add Perplexity API wrapper
   - [ ] Implement RagFlow service
   - [ ] Set up GitHub PR integration
   - [ ] Add OpenAI voice synthesis

4. **UI Enhancements**
   - [ ] Add role indicators
   - [ ] Implement power user controls
   - [ ] Add API feature panels
   - [ ] Create settings conflict resolution UI

## Technical Details

### Settings Storage

#### Local Storage (Unauthenticated)
```typescript
// client/services/SettingsPersistenceService.ts
export class SettingsPersistenceService {
    private readonly LOCAL_STORAGE_KEY = 'logseq_spring_settings';
    // ... implementation
}
```

#### Server Storage (Authenticated)
```rust
// src/models/protected_settings.rs
pub struct ProtectedSettings {
    pub settings: Settings,
    pub users: HashMap<String, NostrUser>,
    // ... implementation
}
```

### Authentication Flow

```typescript
// Authentication check middleware
async function checkAuth(req: Request): Promise<AuthResult> {
    const pubkey = req.headers['X-Nostr-Pubkey'];
    if (!pubkey) return { authenticated: false };
    
    const user = await nostrService.getUser(pubkey);
    return {
        authenticated: true,
        isPowerUser: user.isPowerUser,
        pubkey
    };
}

// Settings sync middleware
async function syncSettings(req: Request): Promise<void> {
    const auth = await checkAuth(req);
    if (!auth.authenticated) return;
    
    if (auth.isPowerUser) {
        // Allow modifications to server settings
        await updateServerSettings(req.body);
    } else {
        // Read-only access to server settings
        throw new Error('Unauthorized to modify settings');
    }
}
```

### API Feature Access

```typescript
// Feature access control
async function checkFeatureAccess(
    pubkey: string,
    feature: 'perplexity' | 'ragflow' | 'github' | 'openai'
): Promise<boolean> {
    const user = await nostrService.getUser(pubkey);
    if (!user.isPowerUser) return false;
    
    // Check specific feature access
    return user.features.includes(feature);
}
```

## Security Considerations

1. **Authentication**
   - Nostr public key verification
   - Session token management
   - Role-based access control

2. **Settings Sync**
   - Validate all settings changes
   - Prevent unauthorized modifications
   - Handle concurrent updates

3. **API Access**
   - Rate limiting
   - API key rotation
   - Usage monitoring

4. **Data Protection**
   - Encrypt sensitive settings
   - Sanitize user inputs
   - Validate all API responses