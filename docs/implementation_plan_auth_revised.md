# Revised Authentication Integration Plan

## Overview
This plan builds upon the existing simplified authentication system while introducing enhanced role-based access control and settings synchronization.

## Phase 1: Enhanced Feature Access System

### 1.1 Extended Environment Configuration
```env
# Existing Configuration
APPROVED_PUBKEYS=npub1...,npub2...
PERPLEXITY_ENABLED_PUBKEYS=npub1...,npub2...
OPENAI_ENABLED_PUBKEYS=npub1...,npub2...
RAGFLOW_ENABLED_PUBKEYS=npub1...,npub2...

# New Configuration
POWER_USER_PUBKEYS=npub1...,npub2...
SETTINGS_SYNC_ENABLED_PUBKEYS=npub1...,npub2...
```

### 1.2 Extended FeatureAccess Implementation
```rust
// src/config/feature_access.rs
pub struct FeatureAccess {
    // Existing fields
    pub approved_pubkeys: Vec<String>,
    pub perplexity_enabled: Vec<String>,
    pub openai_enabled: Vec<String>,
    pub ragflow_enabled: Vec<String>,
    
    // New fields
    pub power_users: Vec<String>,
    pub settings_sync_enabled: Vec<String>,
}

impl FeatureAccess {
    // Existing methods remain unchanged
    
    pub fn is_power_user(&self, pubkey: &str) -> bool {
        self.power_users.contains(&pubkey.to_string())
    }
    
    pub fn can_sync_settings(&self, pubkey: &str) -> bool {
        self.settings_sync_enabled.contains(&pubkey.to_string())
    }
}
```

## Phase 2: Client-Side Authentication Enhancement

### 2.1 Enhanced FeatureService
```typescript
// client/services/FeatureService.ts
export class FeatureService {
    private static instance: FeatureService;
    private enabledFeatures: Set<string> = new Set();
    private userRole: 'basic' | 'power_user' = 'basic';
    
    async initializeFeatures(): Promise<void> {
        // Existing feature checks
        const features = ['perplexity', 'openai', 'ragflow'];
        await Promise.all(
            features.map(async feature => {
                if (await this.checkFeatureAccess(feature)) {
                    this.enabledFeatures.add(feature);
                }
            })
        );
        
        // Check power user status
        const powerUserResponse = await fetch('/api/auth/power-user-status', {
            headers: {
                'X-Nostr-Pubkey': this.getCurrentPubkey()
            }
        });
        
        this.userRole = powerUserResponse.ok ? 'power_user' : 'basic';
    }
    
    isPowerUser(): boolean {
        return this.userRole === 'power_user';
    }
}
```

### 2.2 Settings Sync Integration
```typescript
// client/services/SettingsSyncService.ts
export class SettingsSyncService {
    private featureService: FeatureService;
    
    constructor() {
        this.featureService = FeatureService.getInstance();
    }
    
    async syncSettings(settings: Settings): Promise<void> {
        if (!this.featureService.isPowerUser()) {
            // Store locally only
            await this.storeLocalSettings(settings);
            return;
        }
        
        // Sync with server
        await this.syncWithServer(settings);
    }
    
    private async storeLocalSettings(settings: Settings): Promise<void> {
        // Use existing SettingsPersistenceService
    }
    
    private async syncWithServer(settings: Settings): Promise<void> {
        // Implement server sync for power users
    }
}
```

## Phase 3: Server-Side Implementation

### 3.1 Settings Handler Enhancement
```rust
// src/handlers/settings_handler.rs
pub async fn handle_settings_update(
    req: HttpRequest,
    settings: web::Json<Settings>,
    feature_access: web::Data<FeatureAccess>,
) -> Result<HttpResponse, Error> {
    let pubkey = req.headers()
        .get("X-Nostr-Pubkey")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
        
    if !feature_access.is_power_user(pubkey) {
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Only power users can modify server settings"
        })));
    }
    
    // Process settings update
    update_server_settings(settings.into_inner()).await?;
    
    Ok(HttpResponse::Ok().finish())
}
```

### 3.2 Settings Sync Implementation
```rust
// src/services/settings_sync_service.rs
pub struct SettingsSyncService {
    feature_access: Arc<FeatureAccess>,
}

impl SettingsSyncService {
    pub async fn handle_sync_request(
        &self,
        pubkey: &str,
        settings: Settings,
    ) -> Result<Settings, Error> {
        if !self.feature_access.can_sync_settings(pubkey) {
            return Err(Error::Forbidden);
        }
        
        if self.feature_access.is_power_user(pubkey) {
            // Power users can modify server settings
            self.update_server_settings(settings).await?;
        }
        
        // Return current server settings
        Ok(self.get_server_settings().await?)
    }
}
```

## Phase 4: UI Integration

### 4.1 Role-Based UI Adaptation
```typescript
// client/ui/ModularControlPanel.ts
export class ModularControlPanel {
    private featureService: FeatureService;
    
    constructor() {
        this.featureService = FeatureService.getInstance();
        this.initializePanel();
    }
    
    private async initializePanel(): Promise<void> {
        await this.featureService.initializeFeatures();
        this.updatePanelBasedOnRole();
    }
    
    private updatePanelBasedOnRole(): void {
        const isPowerUser = this.featureService.isPowerUser();
        
        // Update UI elements based on role
        document.querySelectorAll('[data-requires-power-user]')
            .forEach(element => {
                element.style.display = isPowerUser ? 'block' : 'none';
            });
            
        // Update settings controls
        this.updateSettingsControls(isPowerUser);
    }
}
```

## Implementation Steps

1. Update Environment Configuration
   - Add new environment variables
   - Update .env_template
   - Document new configuration options

2. Enhance Feature Access System
   - Extend FeatureAccess struct
   - Add new access check methods
   - Update environment loading

3. Client-Side Implementation
   - Enhance FeatureService
   - Implement SettingsSyncService
   - Update UI components

4. Server-Side Implementation
   - Enhance settings handler
   - Implement sync service
   - Add new API endpoints

5. Testing
   - Unit tests for new functionality
   - Integration tests for sync system
   - End-to-end authentication flow tests

## Security Considerations

1. Environment Variables
   - Secure storage of pubkeys
   - Regular key rotation process
   - Backup and recovery procedures

2. Authentication
   - Validate Nostr signatures
   - Implement rate limiting
   - Add request logging

3. Settings Sync
   - Validate all settings changes
   - Implement conflict resolution
   - Ensure atomic updates

4. Access Control
   - Validate all role-based actions
   - Log access attempts
   - Monitor for suspicious activity

## Success Criteria

1. Existing functionality remains intact
2. Power users can modify server settings
3. Settings sync works reliably
4. UI adapts correctly to user roles
5. Security measures are properly implemented

## Timeline

- Environment Updates: 1 day
- Feature Access Enhancement: 2 days
- Client Implementation: 3 days
- Server Implementation: 3 days
- Testing and Security: 2 days

Total: ~11 days