# Simplified Authentication Plan

## 1. Environment Configuration

Add approved pubkeys to .env:
```env
# Approved Nostr Pubkeys (comma-separated)
APPROVED_PUBKEYS=npub1...,npub2...

# Feature Access Control
PERPLEXITY_ENABLED_PUBKEYS=npub1...,npub2...
OPENAI_ENABLED_PUBKEYS=npub1...,npub2...
RAGFLOW_ENABLED_PUBKEYS=npub1...,npub2...
```

## 2. Server-Side Implementation

### 2.1 Environment Loading
```rust
// src/config.rs
pub struct FeatureAccess {
    pub approved_pubkeys: Vec<String>,
    pub perplexity_enabled: Vec<String>,
    pub openai_enabled: Vec<String>, 
    pub ragflow_enabled: Vec<String>,
}

impl FeatureAccess {
    pub fn from_env() -> Self {
        Self {
            approved_pubkeys: env::var("APPROVED_PUBKEYS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            perplexity_enabled: env::var("PERPLEXITY_ENABLED_PUBKEYS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            openai_enabled: env::var("OPENAI_ENABLED_PUBKEYS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            ragflow_enabled: env::var("RAGFLOW_ENABLED_PUBKEYS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
        }
    }

    pub fn has_access(&self, pubkey: &str) -> bool {
        self.approved_pubkeys.contains(&pubkey.to_string())
    }

    pub fn has_perplexity_access(&self, pubkey: &str) -> bool {
        self.perplexity_enabled.contains(&pubkey.to_string())
    }

    pub fn has_openai_access(&self, pubkey: &str) -> bool {
        self.openai_enabled.contains(&pubkey.to_string())
    }

    pub fn has_ragflow_access(&self, pubkey: &str) -> bool {
        self.ragflow_enabled.contains(&pubkey.to_string())
    }
}
```

### 2.2 API Access Control
```rust
// src/handlers/api_handler.rs
pub async fn check_api_access(
    pubkey: &str,
    feature: &str,
    feature_access: &FeatureAccess,
) -> Result<bool, Error> {
    match feature {
        "perplexity" => Ok(feature_access.has_perplexity_access(pubkey)),
        "openai" => Ok(feature_access.has_openai_access(pubkey)),
        "ragflow" => Ok(feature_access.has_ragflow_access(pubkey)),
        _ => Ok(false)
    }
}

pub async fn handle_api_request(
    req: HttpRequest,
    feature: &str,
    feature_access: &FeatureAccess,
) -> Result<HttpResponse, Error> {
    let pubkey = req.headers()
        .get("X-Nostr-Pubkey")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if !feature_access.has_access(pubkey) {
        return Ok(HttpResponse::Unauthorized().json(json!({
            "error": "Unauthorized access"
        })));
    }

    if !check_api_access(pubkey, feature, feature_access).await? {
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Feature not enabled for this user"
        })));
    }

    // Process API request...
    Ok(HttpResponse::Ok().finish())
}
```

## 3. Client-Side Implementation

### 3.1 Feature Detection
```typescript
// client/services/FeatureService.ts
export class FeatureService {
    private static instance: FeatureService;
    private enabledFeatures: Set<string> = new Set();

    static getInstance(): FeatureService {
        if (!FeatureService.instance) {
            FeatureService.instance = new FeatureService();
        }
        return FeatureService.instance;
    }

    async checkFeatureAccess(feature: string): Promise<boolean> {
        try {
            const response = await fetch(`/api/features/${feature}/access`, {
                headers: {
                    'X-Nostr-Pubkey': this.getCurrentPubkey()
                }
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    async initializeFeatures(): Promise<void> {
        const features = ['perplexity', 'openai', 'ragflow'];
        await Promise.all(
            features.map(async feature => {
                if (await this.checkFeatureAccess(feature)) {
                    this.enabledFeatures.add(feature);
                }
            })
        );
    }

    hasFeatureAccess(feature: string): boolean {
        return this.enabledFeatures.has(feature);
    }
}
```

### 3.2 UI Integration
```typescript
// client/ui/ControlPanel.ts
export class ControlPanel {
    private featureService: FeatureService;

    constructor() {
        this.featureService = FeatureService.getInstance();
        this.initializeFeatures();
    }

    private async initializeFeatures(): Promise<void> {
        await this.featureService.initializeFeatures();
        this.updateUIBasedOnFeatures();
    }

    private updateUIBasedOnFeatures(): void {
        // Show/hide feature controls based on access
        const features = ['perplexity', 'openai', 'ragflow'];
        features.forEach(feature => {
            const featureElement = document.getElementById(`${feature}-controls`);
            if (featureElement) {
                featureElement.style.display = 
                    this.featureService.hasFeatureAccess(feature) ? 'block' : 'none';
            }
        });
    }
}
```

## 4. Implementation Steps

1. Update .env_template with new pubkey configuration sections
2. Implement FeatureAccess in Rust backend
3. Add access control middleware to API handlers
4. Create client-side FeatureService
5. Update ControlPanel to show/hide features based on access
6. Add error handling for unauthorized access attempts
7. Document pubkey configuration process

## 5. Security Considerations

- Store approved pubkeys securely in environment variables
- Validate pubkeys on both client and server side
- Log unauthorized access attempts
- Implement rate limiting per pubkey
- Regular audit of approved pubkeys

This simplified approach provides a straightforward way to gate API features while maintaining security through environment-based configuration.