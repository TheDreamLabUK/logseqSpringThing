import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
// Removed incorrect import: import { requestProvider } from '@getalby/sdk';
import { defaultSettings } from '../features/settings/config/defaultSettings'
import { Settings, SettingsPath } from '../features/settings/config/settings'
import { createLogger, createErrorMetadata } from '../utils/logger'
import { debugState } from '../utils/debugState'
import { deepMerge } from '../utils/deepMerge'
import { convertSnakeToCamelCase } from '../utils/caseConversion'

const logger = createLogger('SettingsStore')

interface SettingsState {
  settings: Settings
  initialized: boolean
  authenticated: boolean
  user: { isPowerUser: boolean; pubkey: string } | null
  subscribers: Map<string, Set<() => void>>

  // Actions
  initialize: () => Promise<Settings>
  setAuthenticated: (authenticated: boolean) => void
  setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void
  get: <T>(path: SettingsPath) => T
  set: <T>(path: SettingsPath, value: T) => void
  subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void
  unsubscribe: (path: SettingsPath, callback: () => void) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: defaultSettings,
      initialized: false,
      authenticated: false,
      user: null,
      subscribers: new Map(),

      initialize: async () => {
        try {
          if (debugState.isEnabled()) {
            logger.info('Initializing settings')
          }

          // Load settings from localStorage via zustand persist
          const currentSettings = get().settings

          // Fetch settings from server if available
          try {
            // Use the endpoint for public/default settings for initial load
            const response = await fetch('/api/user-settings')
            if (response.ok) {
              // Get the server settings and convert from snake_case to camelCase
              const rawServerSettings = await response.json()
              const serverSettings = convertSnakeToCamelCase(rawServerSettings)

              if (debugState.isEnabled()) {
                logger.info('Server settings after case conversion:', { serverSettings })
              }

              // Merge server settings with defaults and current settings using deep merge
              // This ensures all nested objects are properly merged
              const mergedSettings = deepMerge(defaultSettings, currentSettings, serverSettings)

              if (debugState.isEnabled()) {
                logger.info('Deep merged settings:', { mergedSettings })
              }

              set({
                settings: mergedSettings,
                initialized: true
              })

              if (debugState.isEnabled()) {
                logger.info('Settings loaded from server and merged')
              }

              return mergedSettings
            }
          } catch (error) {
            logger.warn('Failed to fetch settings from server:', createErrorMetadata(error))
            // Continue with local settings if server fetch fails
          }

          // Mark as initialized
          set({ initialized: true })

          if (debugState.isEnabled()) {
            logger.info('Settings initialized from local storage')
          }

          return currentSettings
        } catch (error) {
          logger.error('Failed to initialize settings:', createErrorMetadata(error))

          // Fall back to default settings
          set({
            settings: defaultSettings,
            initialized: true
          })

          return defaultSettings
        }
      },

      setAuthenticated: (authenticated: boolean) => set({ authenticated }),

      setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => set({ user }),

      get: <T>(path: SettingsPath): T => {
        const settings = get().settings

        if (!path || path === '') {
          return settings as unknown as T
        }

        // Navigate the settings object using the path
        let current: any = settings
        const pathParts = path.split('.')

        for (const part of pathParts) {
          if (current === undefined || current === null) {
            return undefined as unknown as T
          }
          current = current[part]
        }

        return current as T
      },

      set: <T>(path: SettingsPath, value: T) => {
        set(state => {
          // If setting the entire object
          if (!path || path === '') {
            return { settings: value as unknown as Settings }
          }

          // Create a deep copy of the settings object
          const newSettings = JSON.parse(JSON.stringify(state.settings))

          // Navigate to the correct location and update
          const pathParts = path.split('.')
          let current = newSettings

          // Navigate to the parent of the setting we want to update
          for (let i = 0; i < pathParts.length - 1; i++) {
            const part = pathParts[i]
            if (current[part] === undefined || current[part] === null) {
              // Create the path if it doesn't exist
              current[part] = {}
            }
            current = current[part]
          }

          // Update the value
          const finalPart = pathParts[pathParts.length - 1]
          current[finalPart] = value

          // Return the updated settings
          return { settings: newSettings }
        })

        // Notify subscribers
        const notifySubscribers = async () => {
          const state = get()

          // Build a list of paths to notify
          // e.g. for path 'visualization.bloom.enabled':
          // '', 'visualization', 'visualization.bloom', 'visualization.bloom.enabled'
          const pathsToNotify = ['']
          const pathParts = path.split('.')
          let currentPath = ''

          for (const part of pathParts) {
            currentPath = currentPath ? `${currentPath}.${part}` : part
            pathsToNotify.push(currentPath)
          }

          // Notify subscribers for each path
          for (const notifyPath of pathsToNotify) {
            const callbacks = state.subscribers.get(notifyPath)
            if (callbacks) {
              // Convert Set to Array to avoid TypeScript iteration issues
              Array.from(callbacks).forEach(callback => {
                try {
                  callback()
                } catch (error) {
                  logger.error(`Error in settings subscriber for path ${notifyPath}:`, createErrorMetadata(error))
                }
              })
            }
          }

          // Save to server if appropriate
          if (state.initialized && state.settings.system?.persistSettings !== false) {
            try {
              // Use the endpoint for syncing user-specific settings
              const headers: HeadersInit = {
                'Content-Type': 'application/json',
              };

              // Add Nostr authentication if available
              try {
                // Import nostrAuth dynamically to avoid circular dependencies
                const { nostrAuth } = await import('../services/nostrAuthService')

                if (nostrAuth.isAuthenticated()) {
                  const user = nostrAuth.getCurrentUser()
                  const token = nostrAuth.getSessionToken()

                  if (user && token) {
                    headers['X-Nostr-Pubkey'] = user.pubkey
                    headers['Authorization'] = `Bearer ${token}`
                    logger.info('Using Nostr authentication for settings sync')
                  } else {
                    logger.warn('Nostr auth is authenticated but missing user or token')
                  }
                } else {
                  logger.info('Not authenticated with Nostr, proceeding without auth')
                }
              } catch (error) {
                logger.warn('Error getting Nostr authentication:', createErrorMetadata(error))
                // Proceed without auth header if there's an error
              }

              // Convert settings to camelCase before sending to server
              const settingsToSend = convertSnakeToCamelCase(state.settings)

              if (debugState.isEnabled()) {
                logger.info('Sending settings to server:', { settingsToSend })
              }

              const response = await fetch('/api/user-settings/sync', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(settingsToSend)
              })

              if (!response.ok) {
                throw new Error(`Failed to save settings: ${response.status} ${response.statusText}`)
              }

              if (debugState.isEnabled()) {
                logger.info('Settings saved to server')
              }
            } catch (error) {
              logger.error('Failed to save settings to server:', createErrorMetadata(error))
            }
          }
        }

        // Debounce saving settings
        if (typeof window !== 'undefined') {
          if (window.settingsSaveTimeout) {
            clearTimeout(window.settingsSaveTimeout)
          }
          window.settingsSaveTimeout = setTimeout(notifySubscribers, 300)
        } else {
          // If running server-side, notify immediately
          notifySubscribers()
        }
      },

      subscribe: (path: SettingsPath, callback: () => void, immediate: boolean = true) => {
        set(state => {
          const subscribers = new Map(state.subscribers)

          if (!subscribers.has(path)) {
            subscribers.set(path, new Set())
          }

          subscribers.get(path)!.add(callback)

          return { subscribers }
        })

        // Call callback immediately if requested and initialized
        if (immediate && get().initialized) {
          callback()
        }

        // Return unsubscribe function
        return () => get().unsubscribe(path, callback)
      },

      unsubscribe: (path: SettingsPath, callback: () => void) => {
        set(state => {
          const subscribers = new Map(state.subscribers)

          if (subscribers.has(path)) {
            const callbacks = subscribers.get(path)!
            callbacks.delete(callback)

            if (callbacks.size === 0) {
              subscribers.delete(path)
            }
          }

          return { subscribers }
        })
      }
    }),
    {
      name: 'graph-viz-settings',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        settings: state.settings,
        authenticated: state.authenticated,
        user: state.user
      })
    }
  )
)

// Add to Window interface
declare global {
  interface Window {
    settingsSaveTimeout: ReturnType<typeof setTimeout>;
  }
}
