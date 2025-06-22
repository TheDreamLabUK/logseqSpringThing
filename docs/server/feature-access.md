# Feature Access Control

## Overview

The Feature Access system provides role-based access control (RBAC) for various features and services in LogseqSpringThing. It manages user permissions, feature flags, and API access through environment variables and dynamic registration.

## Architecture

The feature access system is implemented in [`src/config/feature_access.rs`](../../src/config/feature_access.rs) and provides:

1. **Public Key Based Authentication**: Users are identified by their Nostr public keys
2. **Feature Flags**: Enable/disable specific features per user
3. **Role-Based Access**: Special privileges for power users
4. **Dynamic Registration**: New users can be registered with default features

## Core Components

### FeatureAccess Structure

```rust
pub struct FeatureAccess {
    // Base access control
    pub approved_pubkeys: Vec<String>,
    
    // Feature-specific access
    pub perplexity_enabled: Vec<String>,
    pub openai_enabled: Vec<String>,
    pub ragflow_enabled: Vec<String>,
    
    // Role-based access control
    pub power_users: Vec<String>,
    pub settings_sync_enabled: Vec<String>,
}
```

### Environment Variables

The system reads configuration from environment variables:

- `APPROVED_PUBKEYS` - Comma-separated list of approved public keys
- `PERPLEXITY_ENABLED_PUBKEYS` - Users with Perplexity AI access
- `OPENAI_ENABLED_PUBKEYS` - Users with OpenAI (Kokoro) access
- `RAGFLOW_ENABLED_PUBKEYS` - Users with RAGFlow chat access
- `POWER_USER_PUBKEYS` - Users with administrative privileges
- `SETTINGS_SYNC_ENABLED_PUBKEYS` - Users who can sync settings across devices

Example `.env` configuration:
```env
APPROVED_PUBKEYS=pubkey1,pubkey2,pubkey3
PERPLEXITY_ENABLED_PUBKEYS=pubkey1,pubkey2
OPENAI_ENABLED_PUBKEYS=pubkey1,pubkey2,pubkey3
RAGFLOW_ENABLED_PUBKEYS=pubkey1,pubkey2,pubkey3
POWER_USER_PUBKEYS=pubkey1
SETTINGS_SYNC_ENABLED_PUBKEYS=pubkey1,pubkey2
```

## Features and Permissions

### Base Access
- **Controlled by**: `APPROVED_PUBKEYS`
- **Provides**: Basic application access
- **Required for**: All other features

### AI Service Features

#### Perplexity AI
- **Controlled by**: `PERPLEXITY_ENABLED_PUBKEYS`
- **Provides**: Access to Perplexity AI for advanced queries
- **Endpoint**: `/api/perplexity`

#### OpenAI/Kokoro
- **Controlled by**: `OPENAI_ENABLED_PUBKEYS`
- **Provides**: Access to Kokoro text-to-speech service
- **Features**: Voice synthesis, speech options
- **Default for**: New users (auto-granted on registration)

#### RAGFlow Chat
- **Controlled by**: `RAGFLOW_ENABLED_PUBKEYS`
- **Provides**: Access to RAGFlow document-based chat
- **Features**: Contextual chat, document retrieval
- **Default for**: New users (auto-granted on registration)

### System Features

#### Settings Sync
- **Controlled by**: `SETTINGS_SYNC_ENABLED_PUBKEYS`
- **Provides**: Cross-device settings synchronization
- **Auto-granted to**: Power users
- **Endpoints**: `/api/user-settings/sync`

#### Power User Status
- **Controlled by**: `POWER_USER_PUBKEYS`
- **Provides**:
  - All feature access
  - Administrative functions
  - Cache management (`/api/admin/settings/clear-all-cache`)
  - System monitoring
  - Priority support

## API Integration

### Authentication Headers
All protected endpoints require the Nostr public key in headers:
```
X-Nostr-Pubkey: <user_public_key>
```

### Feature Check Endpoints

#### Check Power User Status
```http
GET /api/auth/nostr/power-user-status
Headers: X-Nostr-Pubkey: <pubkey>

Response:
{
  "is_power_user": true
}
```

#### Get Available Features
```http
GET /api/auth/nostr/features
Headers: X-Nostr-Pubkey: <pubkey>

Response:
{
  "features": ["perplexity", "openai", "ragflow", "settings_sync", "power_user"]
}
```

#### Check Specific Feature Access
```http
GET /api/auth/nostr/features/{feature}
Headers: X-Nostr-Pubkey: <pubkey>

Response:
{
  "feature": "perplexity",
  "has_access": true
}
```

## User Registration

New users can be registered dynamically with default features:

```rust
pub fn register_new_user(&mut self, pubkey: &str) -> bool {
    // Adds user to approved_pubkeys
    // Grants RAGFlow access by default
    // Grants OpenAI (Kokoros) access by default
    // Updates the .env file
}
```

### Default Features for New Users
1. Basic application access
2. RAGFlow chat access
3. OpenAI/Kokoro voice service access

### Registration Process
1. User attempts to authenticate with Nostr
2. If pubkey not in `APPROVED_PUBKEYS`, registration is triggered
3. User is added with default features
4. `.env` file is updated to persist changes

## Implementation Details

### Feature Checking
```rust
// Check basic access
if feature_access.has_access(&pubkey) {
    // User has basic access
}

// Check specific feature
if feature_access.has_perplexity_access(&pubkey) {
    // User can use Perplexity AI
}

// Check multiple features
let features = feature_access.get_available_features(&pubkey);
```

### Environment File Updates
The system automatically updates the `.env` file when:
- New users are registered
- Features are granted/revoked programmatically
- This ensures persistence across server restarts

### Access Control in Handlers
Example from the settings handler:
```rust
async fn clear_all_settings_cache(
    req: HttpRequest,
    feature_access: web::Data<FeatureAccess>
) -> Result<HttpResponse, Error> {
    let pubkey = req.headers()
        .get("X-Nostr-Pubkey")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    
    if !feature_access.is_power_user(&pubkey) {
        return Ok(HttpResponse::Forbidden()
            .body("Only power users can clear all settings caches"));
    }
    
    // Proceed with cache clearing...
}
```

## Security Considerations

1. **Public Key Validation**: Always validate public key format
2. **Header Injection**: Sanitize header values
3. **Privilege Escalation**: Power user status cannot be self-assigned
4. **Environment Security**: Protect `.env` file with appropriate permissions
5. **Audit Trail**: Log all access attempts and feature usage

## Best Practices

1. **Principle of Least Privilege**: Grant only necessary features
2. **Regular Audits**: Review user permissions periodically
3. **Feature Grouping**: Consider creating feature bundles for common use cases
4. **Documentation**: Keep feature access documentation up-to-date
5. **Monitoring**: Track feature usage for optimization

## Future Enhancements

1. **Dynamic Feature Management**: Web UI for managing user permissions
2. **Feature Expiration**: Time-limited feature access
3. **Usage Quotas**: Rate limiting per feature
4. **Feature Dependencies**: Automatic dependency resolution
5. **Audit Logging**: Comprehensive access logs
6. **Multi-tenant Support**: Organization-based access control