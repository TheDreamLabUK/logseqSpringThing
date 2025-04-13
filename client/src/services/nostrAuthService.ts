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

export interface NostrEvent {
  id: string;
  pubkey: string;
  content: string;
  sig: string;
  createdAt: number;
  kind: number;
  tags: string[][];
}

class NostrAuthService {
  private static instance: NostrAuthService;
  private sessionToken: string | null = null;
  private currentUser: NostrUser | null = null;

  private constructor() {}

  public static getInstance(): NostrAuthService {
    if (!NostrAuthService.instance) {
      NostrAuthService.instance = new NostrAuthService();
    }
    return NostrAuthService.instance;
  }

  /**
   * Login with Alby or other Nostr provider
   */
  public async login(): Promise<AuthResponse> {
    try {
      // Check if WebLN is available (Alby browser extension)
      if (!window.webln) {
        throw new Error('WebLN provider not found. Please install Alby extension.');
      }

      // Enable WebLN
      await window.webln.enable();
      
      // Get public key
      const pubkey = await window.webln.getPublicKey();
      if (!pubkey) {
        throw new Error('Failed to get public key from WebLN provider');
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
      
      return response;
    } catch (error) {
      logger.error('Nostr login failed:', createErrorMetadata(error));
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
    } catch (error) {
      logger.error('Nostr logout failed:', createErrorMetadata(error));
      throw error;
    }
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

export const nostrAuthService = NostrAuthService.getInstance();
