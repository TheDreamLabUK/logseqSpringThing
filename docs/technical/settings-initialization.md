# Settings Initialization Flow

## Current Issue
The current settings initialization flow has a problem where client-side default settings can overwrite server-side settings during initialization. This happens because:

1. The client starts by loading default settings
2. Then fetches server settings
3. Merges them with defaults (server settings take precedence)
4. But then immediately syncs the merged settings back to server via POST
5. This can potentially overwrite server-specific configurations that shouldn't be changed

## Proposed Solution

### New Initialization Flow
1. First attempt to fetch server settings
   - Make GET request to `/api/settings`
   - If successful, use these settings as the base
   - Validate the server settings

2. Only if server settings fail:
   - Load default settings as fallback
   - Validate default settings
   - Log warning about using defaults

3. Modify syncWithServer behavior:
   - Only sync when there's an explicit settings change (set() method)
   - Don't sync during initialization
   - Add a flag to track if settings came from server

### Code Changes Needed
1. Modify SettingsStore.ts:
   - Reorder initialization to try server first
   - Add serverOrigin flag to track settings source
   - Update syncWithServer to respect server settings

2. Benefits:
   - Preserves server-side configurations
   - Reduces unnecessary API calls
   - Maintains proper settings hierarchy

### Implementation Notes
- The server's settings.yaml should be treated as the source of truth
- Client defaults should only be used when server is unavailable
- All settings changes through the UI will still sync to server
- Initial load should not trigger a sync back to server

This change ensures we maintain proper settings hierarchy while still providing fallback defaults when needed.