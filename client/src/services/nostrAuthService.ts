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
        // Verify the token with the server
        try {
          const response = await apiService.post<AuthResponse>('/auth/nostr/verify', {
            pubkey: JSON.parse(storedUser).pubkey,
            token: storedToken
          });

          if (response) {
            this.sessionToken = storedToken;
            this.currentUser = response.user;
            this.notifyListeners({
              authenticated: true,
              user: {
                isPowerUser: response.user.isPowerUser,
                pubkey: response.user.pubkey
              }
            });
          }
        } catch (error) {
          // Token is invalid, clear storage
          localStorage.removeItem('nostr_session_token');
          localStorage.removeItem('nostr_user');
          logger.warn('Stored session token is invalid, cleared local storage');
        }
      }

      this.initialized = true;
    } catch (error) {
      logger.error('Failed to initialize Nostr auth service:', createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Login with Alby or other Nostr provider
   */
  public async login(): Promise<AuthState> {
    try {
      // Check if WebLN is available (Alby browser extension)
      if (!window.webln) {
        const error = 'WebLN provider not found. Please install Alby extension.';
        this.notifyListeners({
          authenticated: false,
          error
        });
        throw new Error(error);
      }

      // Enable WebLN
      await window.webln.enable();

      // Get public key
      const pubkey = await window.webln.getPublicKey();
      if (!pubkey) {
        const error = 'Failed to get public key from WebLN provider';
        this.notifyListeners({
          authenticated: false,
          error
        });
        throw new Error(error);
      }

      logger.info(`Got pubkey from WebLN: ${pubkey}`);

      // Create a challenge message
      const challenge = `Login to LogseqSpringThing at ${Date.now()}`;

      // Sign the challenge
      const signature = await window.webln.signMessage(challenge);

      // Create a Nostr event
      const event: NostrEvent = {
        id: crypto.randomUUID(),
        pubkey,
        content: challenge,
        sig: signature,
        createdAt: Math.floor(Date.now() / 1000),
        kind: 1,
        tags: []
      };

      // Send the event to the server
      const response = await apiService.post<AuthResponse>('/auth/nostr', event);

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
      if (this.sessionToken) {
        await apiService.delete('/auth/nostr', {
          'Authorization': `Bearer ${this.sessionToken}`
        });
      }

      this.sessionToken = null;
      this.currentUser = null;

      // Clear localStorage
      localStorage.removeItem('nostr_session_token');
      localStorage.removeItem('nostr_user');

      this.notifyListeners({
        authenticated: false
      });
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

// Add WebLN type definition
declare global {
  interface Window {
    webln?: {
      enable: () => Promise<void>;
      getPublicKey: () => Promise<string>;
      signMessage: (message: string) => Promise<string>;
    };
  }
}

export const nostrAuth = NostrAuthService.getInstance();
