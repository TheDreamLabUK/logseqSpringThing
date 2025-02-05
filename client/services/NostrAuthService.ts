import { SettingsEventEmitter, SettingsEventType } from './SettingsEventEmitter';
import { SettingsPersistenceService } from './SettingsPersistenceService';
import { createLogger } from '../core/logger';

const logger = createLogger('NostrAuthService');

/**
 * Represents a Nostr user with their access rights
 */
export interface NostrUser {
    pubkey: string;
    isPowerUser: boolean;
    features: string[];
}

/**
 * Result of an authentication attempt
 */
export interface AuthResult {
    authenticated: boolean;
    user?: NostrUser;
    error?: string;
}

/**
 * Service for handling Nostr authentication and feature access
 */
export class NostrAuthService {
    private static instance: NostrAuthService;
    private currentUser: NostrUser | null = null;
    private eventEmitter: SettingsEventEmitter;
    private settingsPersistence: SettingsPersistenceService;

    private constructor() {
        this.eventEmitter = SettingsEventEmitter.getInstance();
        this.settingsPersistence = SettingsPersistenceService.getInstance();
    }

    /**
     * Get the singleton instance of NostrAuthService
     */
    public static getInstance(): NostrAuthService {
        if (!NostrAuthService.instance) {
            NostrAuthService.instance = new NostrAuthService();
        }
        return NostrAuthService.instance;
    }

    /**
     * Initialize the auth service and check for existing session
     */
    public async initialize(): Promise<void> {
        const storedPubkey = localStorage.getItem('nostr_pubkey');
        if (storedPubkey) {
            await this.checkAuthStatus(storedPubkey);
        }
    }

    /**
     * Attempt to authenticate with a Nostr pubkey
     */
    public async login(): Promise<AuthResult> {
        try {
            // TODO: Implement actual Nostr authentication
            // For now, we'll simulate with a mock pubkey
            const mockPubkey = 'npub1...'; // This will be replaced with actual Nostr auth
            
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Nostr-Pubkey': mockPubkey
                }
            });

            if (!response.ok) {
                throw new Error('Authentication failed');
            }

            const userData = await response.json();
            this.currentUser = {
                pubkey: mockPubkey,
                isPowerUser: userData.isPowerUser,
                features: userData.features
            };

            localStorage.setItem('nostr_pubkey', mockPubkey);
            this.settingsPersistence.setCurrentPubkey(mockPubkey);
            
            this.eventEmitter.emit(SettingsEventType.AUTH_STATE_CHANGED, {
                authState: {
                    isAuthenticated: true,
                    pubkey: mockPubkey
                }
            });

            return {
                authenticated: true,
                user: this.currentUser
            };
        } catch (error) {
            logger.error('Login failed:', error);
            return {
                authenticated: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred'
            };
        }
    }

    /**
     * Log out the current user
     */
    public async logout(): Promise<void> {
        const currentPubkey = this.currentUser?.pubkey;
        
        localStorage.removeItem('nostr_pubkey');
        this.currentUser = null;
        this.settingsPersistence.setCurrentPubkey(null);
        
        this.eventEmitter.emit(SettingsEventType.AUTH_STATE_CHANGED, {
            authState: {
                isAuthenticated: false
            }
        });

        // If user was using server settings, revert to local settings
        if (currentPubkey) {
            await this.settingsPersistence.loadSettings();
        }
    }

    /**
     * Get the current authenticated user
     */
    public getCurrentUser(): NostrUser | null {
        return this.currentUser;
    }

    /**
     * Check if the current user is authenticated
     */
    public isAuthenticated(): boolean {
        return this.currentUser !== null;
    }

    /**
     * Check if the current user is a power user
     */
    public isPowerUser(): boolean {
        return this.currentUser?.isPowerUser || false;
    }

    /**
     * Check if the current user has access to a specific feature
     */
    public hasFeatureAccess(feature: string): boolean {
        return this.currentUser?.features.includes(feature) || false;
    }

    /**
     * Check authentication status with the server
     */
    private async checkAuthStatus(pubkey: string): Promise<void> {
        try {
            const response = await fetch('/api/auth/status', {
                headers: {
                    'X-Nostr-Pubkey': pubkey
                }
            });

            if (!response.ok) {
                throw new Error('Authentication check failed');
            }

            const userData = await response.json();
            this.currentUser = {
                pubkey,
                isPowerUser: userData.isPowerUser,
                features: userData.features
            };

            this.settingsPersistence.setCurrentPubkey(pubkey);
            
            this.eventEmitter.emit(SettingsEventType.AUTH_STATE_CHANGED, {
                authState: {
                    isAuthenticated: true,
                    pubkey
                }
            });
        } catch (error) {
            logger.error('Auth check failed:', error);
            await this.logout();
        }
    }

    /**
     * Subscribe to authentication state changes
     */
    public onAuthStateChanged(callback: (state: { authenticated: boolean; user?: NostrUser }) => void): () => void {
        return this.eventEmitter.on(SettingsEventType.AUTH_STATE_CHANGED, (data) => {
            callback({
                authenticated: data.authState?.isAuthenticated || false,
                user: this.currentUser || undefined
            });
        });
    }
}

// Export singleton instance
export const nostrAuth = NostrAuthService.getInstance();