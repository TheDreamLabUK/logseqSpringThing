# Settings Initialization Flow

## Overview
The settings initialization process handles both server-side settings and user-specific settings through Nostr authentication. This document outlines the initialization flow and how different settings sources are prioritized.

## Settings Hierarchy

1. Server Settings (Highest Priority)
   - Defined in settings.yaml
   - Contains system-wide configurations
   - Protected settings that shouldn't be overwritten

2. Nostr User Settings
   - User-specific settings stored with Nostr authentication
   - Persisted per user pubkey
   - Synced across sessions

3. Local Default Settings (Lowest Priority)
   - Defined in defaultSettings.ts
   - Used as fallback when server/Nostr settings unavailable
   - Basic configuration for new users

## Initialization Flow

1. First attempt to fetch server settings
   - Make GET request to `/api/settings`
   - If successful, use these settings as the base
   - Validate the server settings

2. If user is authenticated via Nostr:
   - Fetch user-specific settings using pubkey
   - Merge with server settings (user settings take precedence for customizable options)
   - Preserve protected server settings

3. If server settings fail or user not authenticated:
   - Load default settings as fallback
   - Validate default settings
   - Log warning about using defaults

4. Settings Persistence:
   - Server settings remain in settings.yaml
   - User settings stored with Nostr credentials
   - Local settings cached in browser storage

## Implementation Details

### SettingsStore

```typescript
class SettingsStore {
    private settings: Settings;
    private serverOrigin: boolean;
    private nostrAuth: NostrAuthService;
    private settingsPersistence: SettingsPersistenceService;

    async initialize() {
        try {
            // 1. Try server settings first
            const serverSettings = await this.fetchServerSettings();
            this.settings = serverSettings;
            this.serverOrigin = true;

            // 2. If authenticated, merge with user settings
            if (this.nostrAuth.isAuthenticated()) {
                const userSettings = await this.settingsPersistence.loadUserSettings(
                    this.nostrAuth.getCurrentUser()?.pubkey
                );
                this.mergeSettings(userSettings, true);
            }
        } catch (error) {
            // 3. Fallback to defaults
            this.settings = defaultSettings;
            this.serverOrigin = false;
            logger.warn('Using default settings:', error);
        }

        // Validate final settings
        this.validate(this.settings);
    }

    private mergeSettings(newSettings: Partial<Settings>, isUserSettings: boolean) {
        if (isUserSettings) {
            // Merge user customizable settings only
            this.settings = {
                ...this.settings,
                ...this.filterCustomizableSettings(newSettings)
            };
        } else {
            // Full merge for non-user settings
            this.settings = {
                ...this.settings,
                ...newSettings
            };
        }
    }
}
```

### NostrAuthService Integration

```typescript
class NostrAuthService {
    async login(): Promise<AuthResult> {
        const result = await this.authenticate();
        if (result.authenticated) {
            // Load user settings after successful auth
            await this.settingsPersistence.loadUserSettings(result.user.pubkey);
            this.eventEmitter.emit(SettingsEventType.AUTH_STATE_CHANGED, {
                authState: { isAuthenticated: true, pubkey: result.user.pubkey }
            });
        }
        return result;
    }

    async logout(): Promise<void> {
        const currentPubkey = this.currentUser?.pubkey;
        if (currentPubkey) {
            // Save user settings before logout
            await this.settingsPersistence.saveUserSettings(currentPubkey);
        }
        // Clear user settings and revert to defaults
        await this.settingsPersistence.loadSettings();
    }
}
```

## Benefits

1. Clear Settings Hierarchy
   - Server settings preserved for critical configurations
   - User preferences maintained across sessions
   - Reliable fallback to defaults

2. Improved User Experience
   - Settings persist across devices through Nostr
   - Quick restoration of preferences on login
   - Protected settings prevent accidental overwrites

3. Technical Advantages
   - Reduced unnecessary API calls
   - Better error handling and fallbacks
   - Clear separation of concerns

## Implementation Notes

- Server's settings.yaml is the source of truth for system settings
- User settings only modify allowed customization options
- All UI-based settings changes sync to both server and Nostr storage
- Initial load prevents unnecessary syncing
- Clear logging for debugging settings flow

This architecture ensures proper settings management while providing flexibility for user customization and reliable persistence through Nostr authentication.