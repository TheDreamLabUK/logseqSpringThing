import { apiService } from './api';
import { createLogger, createErrorMetadata } from '../utils/logger';
import { WebLNProvider } from '@getalby/sdk';
import { Event, UnsignedEvent } from 'nostr-tools';

const logger = createLogger('NostrAuthService');

// Keep existing interfaces, adding VerifyResponse and AuthEventPayload

export interface NostrUser {
  pubkey: string; // hex pubkey
  npub?: string; // npub format
  isPowerUser: boolean;
  // Add fields that might be stored locally but not sent in every API response
  api_keys?: { // Optional, as it's not part of AuthResponse user DTO
      perplexity?: string | null;
      openai?: string | null;
      ragflow?: string | null;
  };
  last_seen?: number; // Optional, managed server-side mostly
  session_token?: string | null; // Keep track locally
}

export interface AuthResponse {
  user: { // Structure from server DTO
    pubkey: string;
    npub?: string;
    isPowerUser: boolean;
  };
  token: string;
  expiresAt: number;
  features: string[];
}

export interface AuthState {
  authenticated: boolean;
  user?: { // Simplified state for UI
    isPowerUser: boolean;
    pubkey: string; // hex pubkey
    npub?: string; // npub format
  };
  error?: string;
}

// Interface matching the server's AuthEvent struct (used for POST /auth/nostr)
export interface AuthEventPayload {
  id: string;
  pubkey: string;
  content: string;
  sig: string;
  created_at: number; // Use number for timestamp
  kind: number;
  tags: string[][];
}

// Define the structure for the /verify endpoint response based on server code
export interface VerifyResponse {
  valid: boolean;
  user?: {
    pubkey: string;
    npub?: string;
    isPowerUser: boolean;
  };
  features: string[];
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
    if (this.initialized) return;

    try {
      const storedToken = localStorage.getItem('nostr_session_token');
      const storedUserJson = localStorage.getItem('nostr_user');

      if (storedToken && storedUserJson) {
        let storedUser: NostrUser | null = null;
        try {
          storedUser = JSON.parse(storedUserJson);
        } catch (parseError) {
            logger.error('Failed to parse stored user data:', createErrorMetadata(parseError));
            localStorage.removeItem('nostr_session_token');
            localStorage.removeItem('nostr_user');
        }

        if (storedUser) {
            logger.info(`Verifying stored session for pubkey: ${storedUser.pubkey}`);
            try {
              // Type assertion needed as apiService.post returns Promise<T>
              const verificationResponse = await apiService.post<VerifyResponse>('/auth/nostr/verify', {
                pubkey: storedUser.pubkey, // Use hex pubkey from stored user
                token: storedToken
              });

              if (verificationResponse.valid && verificationResponse.user) {
                // Restore session using verificationResponse data
                this.sessionToken = storedToken;
                // Map VerifyResponse.user to NostrUser type
                this.currentUser = {
                    pubkey: verificationResponse.user.pubkey,
                    npub: verificationResponse.user.npub || undefined,
                    isPowerUser: verificationResponse.user.isPowerUser,
                    // Restore potentially missing fields from local storage if needed, or use defaults
                    api_keys: storedUser.api_keys || undefined, // Keep locally stored API keys if present
                    last_seen: Math.floor(Date.now() / 1000), // Update last_seen on successful verify
                    session_token: storedToken
                };

                // Update local storage with potentially refreshed user data (e.g., isPowerUser status)
                // Only store essential user info, not the full internal state object
                const userToStore = {
                    pubkey: this.currentUser.pubkey,
                    npub: this.currentUser.npub,
                    isPowerUser: this.currentUser.isPowerUser,
                    api_keys: this.currentUser.api_keys // Persist API keys if they were loaded
                };
                localStorage.setItem('nostr_user', JSON.stringify(userToStore));


                this.notifyListeners({
                    authenticated: true,
                    user: {
                        isPowerUser: this.currentUser.isPowerUser,
                        pubkey: this.currentUser.pubkey,
                        npub: this.currentUser.npub
                    }
                 });
                logger.info('Restored and verified session from local storage');
              } else {
                // Token invalid or user mismatch, clear storage
                localStorage.removeItem('nostr_session_token');
                localStorage.removeItem('nostr_user');
                logger.warn('Stored session token is invalid or user mismatch, cleared local storage');
                this.notifyListeners({ authenticated: false });
              }
            } catch (error) {
               // Handle API error during verification, clear storage
               localStorage.removeItem('nostr_session_token');
               localStorage.removeItem('nostr_user');
               logger.error('Failed to verify stored session:', createErrorMetadata(error));
               this.notifyListeners({ authenticated: false, error: 'Session verification failed' });
            }
        }
      } else {
        logger.info('No stored session found');
        this.notifyListeners({ authenticated: false });
      }

      this.initialized = true;
    } catch (error) {
      logger.error('Failed to initialize Nostr auth service:', createErrorMetadata(error));
      this.notifyListeners({ authenticated: false, error: 'Initialization failed' });
      // Do not re-throw, allow app to load in logged-out state
    }
  }

  /**
   * Login with Nostr
   */
  public async login(): Promise<AuthState> {
    let authState: AuthState;
    try {
      // 1. Use Alby SDK to interact with NIP-07 wallet
      const alby = new WebLNProvider();
      // Ensure the user is available (this might trigger the Alby prompt)
      const pubkey = await alby.getPublicKey();

      if (!pubkey) {
        throw new Error('Could not get public key from Alby extension.');
      }
      logger.info(`Got pubkey from Alby: ${pubkey}`);

      // 2. Construct NIP-42 Authentication Event (Kind 22242)
      const challenge = crypto.randomUUID();
      const relayUrl = 'wss://relay.damus.io';

      const unsignedEvent: UnsignedEvent = {
        kind: 22242,
        created_at: Math.floor(Date.now() / 1000),
        tags: [
          ['relay', relayUrl],
          ['challenge', challenge]
        ],
        pubkey: pubkey,
        content: 'Authenticate to Logseq Spring Thing'
      };

      // 3. Sign the event using Alby
      logger.debug('Requesting signature from Alby for event:', unsignedEvent);
      const signedEvent = await alby.signEvent(unsignedEvent) as Event;
      logger.debug('Event signed successfully by Alby.');

      // Map to the AuthEventPayload structure expected by the backend
      const eventPayload: AuthEventPayload = {
        id: signedEvent.id,
        pubkey: signedEvent.pubkey,
        content: signedEvent.content,
        sig: signedEvent.sig,
        created_at: signedEvent.created_at,
        kind: signedEvent.kind,
        tags: signedEvent.tags,
      };


      // 4. Send the signed event to the backend API
      logger.info(`Sending auth event to /api/auth/nostr for pubkey: ${pubkey}`);
      const response = await apiService.post<AuthResponse>('/auth/nostr', eventPayload);
      logger.info(`Auth successful for pubkey: ${response.user.pubkey}`);

      // 5. Store session and update state
      this.sessionToken = response.token;
      // Map the response DTO to the internal NostrUser type
      this.currentUser = {
        pubkey: response.user.pubkey,
        npub: response.user.npub,
        isPowerUser: response.user.isPowerUser,
        session_token: response.token,
        last_seen: Math.floor(Date.now() / 1000)
        // api_keys are not returned by login, keep undefined or default
      };

      // Store essential user info, not the full internal state object
      const userToStore = {
          pubkey: this.currentUser.pubkey,
          npub: this.currentUser.npub,
          isPowerUser: this.currentUser.isPowerUser,
          // Do not store session_token or last_seen in user object in local storage
      };
      localStorage.setItem('nostr_session_token', response.token);
      localStorage.setItem('nostr_user', JSON.stringify(userToStore));

      authState = {
        authenticated: true,
        user: {
          isPowerUser: this.currentUser.isPowerUser,
          pubkey: this.currentUser.pubkey,
          npub: this.currentUser.npub
        }
      };

      this.notifyListeners(authState);
      return authState;

    } catch (error: any) {
      logger.error('Nostr login failed:', createErrorMetadata(error));

      let errorMessage = 'Login failed';
      if (error?.response?.data?.error) { // Check for backend error structure
          errorMessage = error.response.data.error;
      } else if (error?.message) {
          errorMessage = error.message;
      } else if (typeof error === 'string') {
          errorMessage = error;
      }
      // Check for specific Alby/Nostr errors if possible
      if (errorMessage.includes('extension rejected') || errorMessage.includes('User rejected')) {
          errorMessage = 'Login request rejected in Alby extension.';
      } else if (errorMessage.includes('401') || errorMessage.includes('Invalid signature')) { // Check for specific backend error message
          errorMessage = 'Authentication failed: Invalid signature or credentials.';
      } else if (errorMessage.includes('Could not get public key')) {
           errorMessage = 'Failed to get public key. Is Alby extension installed and unlocked?';
      }


      authState = {
        authenticated: false,
        error: errorMessage
      };

      this.notifyListeners(authState);
      // Re-throw the error so UI components can potentially handle it too
      throw new Error(errorMessage);
    }
  }

  /**
   * Logout from Nostr
   */
  public async logout(): Promise<void> {
    logger.info('Attempting logout...');
    const token = this.sessionToken;
    const user = this.currentUser;

    // Clear local state immediately for faster UI update
    this.sessionToken = null;
    this.currentUser = null;
    localStorage.removeItem('nostr_session_token');
    localStorage.removeItem('nostr_user');
    this.notifyListeners({ authenticated: false }); // Notify UI immediately

    if (token && user) {
      try {
        logger.info(`Calling server logout for pubkey: ${user.pubkey}`);
        // Server expects DELETE with pubkey and token in body
        // Assuming apiService.delete can handle a body, otherwise backend needs adjustment
         // Pass data in the body for DELETE request
         await apiService.delete<any>('/auth/nostr', {
             pubkey: user.pubkey,
             token: token
         });
        logger.info('Server logout successful');
      } catch (error) {
        // Log the error but don't re-throw, as client-side logout is already done
        logger.error('Server logout call failed:', createErrorMetadata(error));
        // Optionally notify listeners about the server error?
        // this.notifyListeners({ authenticated: false, error: 'Server logout failed' });
      }
    } else {
        logger.warn('Logout called but no active session found locally.');
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
          pubkey: this.currentUser.pubkey,
          npub: this.currentUser.npub
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
