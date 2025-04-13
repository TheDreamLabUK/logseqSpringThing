import { nostrAuth } from '../../services/nostrAuthService';
import { createLogger, createErrorMetadata } from '../../utils/logger';

const logger = createLogger('initAuth');

/**
 * Initialize the authentication system
 */
export async function initializeAuth(): Promise<void> {
  try {
    // Initialize Nostr auth service
    await nostrAuth.initialize();
    
    logger.info('Auth system initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize auth system:', createErrorMetadata(error));
    throw error;
  }
}
