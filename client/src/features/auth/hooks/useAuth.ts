import { useState } from 'react';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger, createErrorMetadata } from '../../../utils/logger';
import { nostrAuthService } from '../../../services/nostrAuthService';

const logger = createLogger('useAuth');

const useAuth = () => {
  const { setAuthenticated, setUser, authenticated, user } = useSettingsStore();
  const [authError, setAuthError] = useState<string | null>(null);

  const login = async () => {
    try {
      setAuthError(null);

      // Use the Nostr auth service to login
      const authResponse = await nostrAuthService.login();

      // Update the settings store with the user info
      setAuthenticated(true);
      setUser({
        isPowerUser: authResponse.user.isPowerUser,
        pubkey: authResponse.user.pubkey
      });

      logger.info('Login successful');
    } catch (error) {
      logger.error('Login failed:', createErrorMetadata(error));
      setAuthError(error instanceof Error ? error.message : 'Login failed');
    }
  };

  const logout = async () => {
    try {
      setAuthError(null);

      // Use the Nostr auth service to logout
      await nostrAuthService.logout();

      // Update the settings store
      setAuthenticated(false);
      setUser(null);

      logger.info('Logout successful');
    } catch (error) {
      logger.error('Logout failed:', createErrorMetadata(error));
      setAuthError(error instanceof Error ? error.message : 'Logout failed');
    }
  };

  return {
    authenticated,
    user,
    authError,
    login,
    logout,
  };
};

export default useAuth;