# Technical Implementation Details for Nostr Integration

## 1. Client-Side Authentication Layer

### 1.1 NostrAuthService
```typescript
// client/services/NostrAuthService.ts
export class NostrAuthService {
  private static instance: NostrAuthService;
  private currentUser: NostrUser | null = null;
  private authState: AuthState = {
    isAuthenticated: false,
    isPowerUser: false,
    sessionToken: null,
    expiresAt: null
  };

  // Singleton pattern
  static getInstance(): NostrAuthService {
    if (!NostrAuthService.instance) {
      NostrAuthService.instance = new NostrAuthService();
    }
    return NostrAuthService.instance;
  }

  async login(authEvent: AuthEvent): Promise<void> {
    const response = await fetch('/auth/nostr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(authEvent)
    });
    
    if (response.ok) {
      const { user, token, expires_at } = await response.json();
      this.setAuthState(user, token, expires_at);
      this.startSessionRefreshTimer();
    }
  }

  private setAuthState(user: NostrUser, token: string, expiresAt: number): void {
    this.currentUser = user;
    this.authState = {
      isAuthenticated: true,
      isPowerUser: user.is_power_user,
      sessionToken: token,
      expiresAt
    };
    localStorage.setItem('nostr_auth', JSON.stringify(this.authState));
  }

  private startSessionRefreshTimer(): void {
    // Refresh 5 minutes before expiration
    const refreshTime = (this.authState.expiresAt! - Date.now()) - (5 * 60 * 1000);
    setTimeout(() => this.refreshSession(), refreshTime);
  }
}
```

### 1.2 Settings Persistence Service
```typescript
// client/services/SettingsPersistenceService.ts
export class SettingsPersistenceService {
  private static instance: SettingsPersistenceService;
  private nostrAuth: NostrAuthService;

  constructor() {
    this.nostrAuth = NostrAuthService.getInstance();
  }

  async saveSettings(settings: LayoutConfig): Promise<void> {
    // Always save locally
    localStorage.setItem('panel_settings', JSON.stringify(settings));

    // If authenticated, sync to server
    if (this.nostrAuth.isAuthenticated()) {
      await this.syncToServer(settings);
    }
  }

  private async syncToServer(settings: LayoutConfig): Promise<void> {
    const response = await fetch('/api/settings/sync', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.nostrAuth.getSessionToken()}`
      },
      body: JSON.stringify(settings)
    });

    if (!response.ok) {
      throw new Error('Failed to sync settings to server');
    }
  }
}
```

## 2. Enhanced Control Panel Components

### 2.1 AuthenticatedControlPanel
```typescript
// client/components/AuthenticatedControlPanel.ts
export class AuthenticatedControlPanel extends ControlPanel {
  private nostrAuth: NostrAuthService;
  private settingsPersistence: SettingsPersistenceService;

  constructor() {
    super();
    this.nostrAuth = NostrAuthService.getInstance();
    this.settingsPersistence = new SettingsPersistenceService();
    this.initializeAuthUI();
  }

  private initializeAuthUI(): void {
    const authSection = document.createElement('div');
    authSection.className = 'auth-section';
    
    if (this.nostrAuth.isAuthenticated()) {
      this.renderAuthenticatedUI(authSection);
    } else {
      this.renderLoginUI(authSection);
    }

    this.container.insertBefore(authSection, this.container.firstChild);
  }

  protected override async saveSettings(): Promise<void> {
    const settings = this.gatherCurrentSettings();
    await this.settingsPersistence.saveSettings(settings);
  }
}
```

### 2.2 PowerUserFeatures
```typescript
// client/components/PowerUserFeatures.ts
export class PowerUserFeatures {
  private apiKeyManager: HTMLElement;
  private perplexitySection: HTMLElement;

  constructor(container: HTMLElement) {
    this.initializeAPIKeyManager(container);
    this.initializePerplexitySection(container);
  }

  private async initializeAPIKeyManager(container: HTMLElement): Promise<void> {
    this.apiKeyManager = document.createElement('div');
    this.apiKeyManager.className = 'api-key-manager';
    
    const keys = await this.fetchCurrentAPIKeys();
    this.renderAPIKeyInputs(keys);
    
    container.appendChild(this.apiKeyManager);
  }

  private async fetchCurrentAPIKeys(): Promise<ApiKeys> {
    const response = await fetch('/auth/nostr/api-keys', {
      headers: {
        'Authorization': `Bearer ${NostrAuthService.getInstance().getSessionToken()}`
      }
    });
    return response.json();
  }
}
```

## 3. Settings Synchronization

### 3.1 Sync Manager
```typescript
// client/services/SettingsSyncManager.ts
export class SettingsSyncManager {
  private static readonly SYNC_INTERVAL = 5 * 60 * 1000; // 5 minutes
  private syncTimer: number | null = null;

  startAutoSync(): void {
    if (this.syncTimer) return;
    
    this.syncTimer = window.setInterval(() => {
      this.performSync();
    }, SettingsSyncManager.SYNC_INTERVAL);
  }

  private async performSync(): Promise<void> {
    const localSettings = this.loadLocalSettings();
    const serverSettings = await this.fetchServerSettings();
    
    const mergedSettings = this.mergeSettings(localSettings, serverSettings);
    await this.saveSettings(mergedSettings);
  }

  private mergeSettings(local: LayoutConfig, server: LayoutConfig): LayoutConfig {
    // Prefer server settings for authenticated users, but keep local-only changes
    return {
      ...local,
      ...server,
      sections: this.mergeSections(local.sections, server.sections)
    };
  }
}
```

## 4. Security Implementation

### 4.1 Token Management
```typescript
// client/services/TokenManager.ts
export class TokenManager {
  private static readonly TOKEN_KEY = 'nostr_session_token';
  private static readonly EXPIRY_KEY = 'nostr_token_expiry';

  static storeToken(token: string, expiresAt: number): void {
    // Use secure storage methods when available
    if (window.crypto && window.crypto.subtle) {
      this.securelyStoreToken(token, expiresAt);
    } else {
      // Fallback to localStorage with encryption
      this.encryptAndStore(token, expiresAt);
    }
  }

  private static async securelyStoreToken(token: string, expiresAt: number): Promise<void> {
    const encoder = new TextEncoder();
    const tokenData = encoder.encode(token);
    
    const key = await window.crypto.subtle.generateKey(
      { name: "AES-GCM", length: 256 },
      true,
      ["encrypt", "decrypt"]
    );

    const encryptedToken = await window.crypto.subtle.encrypt(
      { name: "AES-GCM", iv: window.crypto.getRandomValues(new Uint8Array(12)) },
      key,
      tokenData
    );

    localStorage.setItem(this.TOKEN_KEY, btoa(String.fromCharCode(...new Uint8Array(encryptedToken))));
    localStorage.setItem(this.EXPIRY_KEY, expiresAt.toString());
  }
}
```

## 5. Migration Strategy

### 5.1 Settings Migration
```typescript
// client/services/SettingsMigration.ts
export class SettingsMigration {
  static async migrateToNostr(): Promise<void> {
    // 1. Gather existing settings
    const existingSettings = this.gatherExistingSettings();
    
    // 2. Create new settings structure
    const newSettings = this.createNostrCompatibleSettings(existingSettings);
    
    // 3. Backup existing settings
    this.backupSettings(existingSettings);
    
    // 4. Apply new settings
    await this.applyNewSettings(newSettings);
  }

  private static gatherExistingSettings(): any {
    return {
      local: JSON.parse(localStorage.getItem('panel_settings') || '{}'),
      sessionStorage: JSON.parse(sessionStorage.getItem('temp_settings') || '{}')
    };
  }
}
```

This technical implementation provides a robust foundation for integrating Nostr authentication with the modular control panel, ensuring secure and efficient handling of user settings and API access.