import { apiService } from './api';
import { createLogger, createErrorMetadata } from '../utils/logger';

const logger = createLogger('NostrAuthService');

export interface NostrUser {
  pubkey: string;
  npub?: string;
  isPowerUser: boolean;
}

export interface AuthResponse {
  user: NostrUser;
  token: string;
  expiresAt: number;
  features: string[];
}

export interface AuthState {
  authenticated: boolean;
  user?: {
    isPowerUser: boolean;
    pubkey: string;
  };
  error?: string;
}

export interface NostrEvent {
  id: string;
  pubkey: string;
  content: string;
  sig: string;
  createdAt: number;
  kind: number;
  tags: string[][];
}

type AuthStateListener = (state: AuthState) => void;

class NostrAuthService {
  private static instance: NostrAuthService;
  private sessionToken: string | null = null;
  private currentUser: NostrUser | null = null;
  private authStateListeners: AuthStateListener[] = [];
  private initialized = false;

  private constructor() {}

  public static getInstance(): NostrAuthService {
    if (!NostrAuthService.instance) {
      NostrAuthService.instance = new NostrAuthService();
    }
    return NostrAuthService.instance;
  }

  /**
   * Initialize the auth service
   */
  public async initialize(): Promise<void> {
    try {
      // Check for stored session token
      const storedToken = localStorage.getItem('nostr_session_token');
      const storedUser = localStorage.getItem('nostr_user');

      if (storedToken && storedUser) {
        try {
          // In a real implementation, you would verify the token with the server
          // For demonstration, we'll just use the stored values
          const parsedUser = JSON.parse(storedUser);

          this.sessionToken = storedToken;
          this.currentUser = parsedUser;

          this.notifyListeners({
            authenticated: true,
            user: {
              isPowerUser: parsedUser.isPowerUser,
              pubkey: parsedUser.pubkey
            }
          });

          logger.info('Restored session from local storage');
        } catch (error) {
          // Token is invalid, clear storage
          localStorage.removeItem('nostr_session_token');
          localStorage.removeItem('nostr_user');
          logger.warn('Stored session data is invalid, cleared local storage');
        }
      } else {
        logger.info('No stored session found');
      }

      this.initialized = true;
    } catch (error) {
      logger.error('Failed to initialize Nostr auth service:', createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Login with Nostr
   */
  public async login(): Promise<AuthState> {
    try {
      // For simplicity, we'll use a mock pubkey for demonstration
      // In a real implementation, you would get this from your Nostr provider
      const pubkey = 'npub1abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklm';

      logger.info(`Using pubkey: ${pubkey}`);

      // In a real implementation, you would create a proper Nostr event
      // and sign it with the user's private key

      // For now, we'll simulate a successful authentication response
      const mockResponse: AuthResponse = {
        user: {
          pubkey,
          npub: pubkey,
          isPowerUser: true
        },
        token: 'mock-session-token-' + Date.now(),
        expiresAt: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
        features: ['basic', 'advanced']
      };

      // In a real implementation, you would send the event to the server
      // const response = await apiService.post<AuthResponse>('/auth/nostr', event);

      // For demonstration, we'll use the mock response
      const response = mockResponse;

      // Store the session token and user
      this.sessionToken = response.token;
      this.currentUser = response.user;

      // Store in localStorage for persistence
      localStorage.setItem('nostr_session_token', response.token);
      localStorage.setItem('nostr_user', JSON.stringify(response.user));

      const authState: AuthState = {
        authenticated: true,
        user: {
          isPowerUser: response.user.isPowerUser,
          pubkey: response.user.pubkey
        }
      };

      this.notifyListeners(authState);
      return authState;
    } catch (error) {
      logger.error('Nostr login failed:', createErrorMetadata(error));

      const authState: AuthState = {
        authenticated: false,
        error: error instanceof Error ? error.message : 'Login failed'
      };

      this.notifyListeners(authState);
      throw error;
    }
  }

  /**
   * Logout from Nostr
   */
  public async logout(): Promise<void> {
    try {
      // In a real implementation, you would call the server to invalidate the session
      // if (this.sessionToken) {
      //   await apiService.delete('/auth/nostr', {
      //     'Authorization': `Bearer ${this.sessionToken}`
      //   });
      // }

      // For demonstration, we'll just clear the local state
      this.sessionToken = null;
      this.currentUser = null;

      // Clear localStorage
      localStorage.removeItem('nostr_session_token');
      localStorage.removeItem('nostr_user');

      this.notifyListeners({
        authenticated: false
      });

      logger.info('Logged out successfully');
    } catch (error) {
      logger.error('Nostr logout failed:', createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Register a listener for auth state changes
   */
  public onAuthStateChanged(listener: AuthStateListener): () => void {
    this.authStateListeners.push(listener);

    // Immediately notify with current state
    if (this.initialized) {
      listener({
        authenticated: this.isAuthenticated(),
        user: this.currentUser ? {
          isPowerUser: this.currentUser.isPowerUser,
          pubkey: this.currentUser.pubkey
        } : undefined
      });
    }

    // Return unsubscribe function
    return () => {
      this.authStateListeners = this.authStateListeners.filter(l => l !== listener);
    };
  }

  /**
   * Notify all listeners of auth state changes
   */
  private notifyListeners(state: AuthState): void {
    this.authStateListeners.forEach(listener => {
      try {
        listener(state);
      } catch (error) {
        logger.error('Error in auth state listener:', createErrorMetadata(error));
      }
    });
  }

  /**
   * Get the current user
   */
  public getCurrentUser(): NostrUser | null {
    return this.currentUser;
  }

  /**
   * Get the session token
   */
  public getSessionToken(): string | null {
    return this.sessionToken;
  }

  /**
   * Check if the user is authenticated
   */
  public isAuthenticated(): boolean {
    return !!this.sessionToken && !!this.currentUser;
  }
}

// No global type definitions needed

export const nostrAuth = NostrAuthService.getInstance();
