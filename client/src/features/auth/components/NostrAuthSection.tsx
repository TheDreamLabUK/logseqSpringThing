import React, { useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../../ui/Card'
import AuthUIHandler from './AuthUIHandler'
import { initializeAuth } from '../initAuth'
import { createLogger, createErrorMetadata } from '../../../utils/logger'

const logger = createLogger('NostrAuthSection');

const NostrAuthSection: React.FC = () => {
  useEffect(() => {
    // Initialize auth system when component mounts
    initializeAuth().catch(error => {
      logger.error('Failed to initialize auth system:', createErrorMetadata(error));
    });
  }, []);

  return (
    // Explicitly set dark background and text for the card to ensure theme consistency
    <Card className="bg-card text-card-foreground border-border">
      <CardHeader>
        <CardTitle>Nostr Authentication</CardTitle>
        <CardDescription>Authenticate with your Nostr key to unlock advanced features.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col space-y-2">
        <AuthUIHandler />
      </CardContent>
    </Card>
  )
}

export default NostrAuthSection
