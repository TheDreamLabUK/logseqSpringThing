import React, { useEffect, useState } from 'react';
import { srOnlyClasses } from '../utils/accessibility';

interface LiveRegionProps {
  message: string;
  priority?: 'polite' | 'assertive';
  clearAfter?: number; // milliseconds
}

/**
 * LiveRegion component for screen reader announcements
 * Provides real-time updates to assistive technologies
 */
export function LiveRegion({ message, priority = 'polite', clearAfter = 5000 }: LiveRegionProps) {
  const [currentMessage, setCurrentMessage] = useState(message);

  useEffect(() => {
    setCurrentMessage(message);

    if (clearAfter && message) {
      const timer = setTimeout(() => {
        setCurrentMessage('');
      }, clearAfter);

      return () => clearTimeout(timer);
    }
  }, [message, clearAfter]);

  return (
    <div
      role="status"
      aria-live={priority}
      aria-atomic="true"
      className={srOnlyClasses}
    >
      {currentMessage}
    </div>
  );
}

/**
 * Hook for using live regions in components
 */
export function useLiveRegion() {
  const [message, setMessage] = useState('');
  const [priority, setPriority] = useState<'polite' | 'assertive'>('polite');

  const announce = (newMessage: string, newPriority: 'polite' | 'assertive' = 'polite') => {
    // Clear and reset to ensure screen readers pick up the change
    setMessage('');
    setPriority(newPriority);
    
    setTimeout(() => {
      setMessage(newMessage);
    }, 100);
  };

  return {
    message,
    priority,
    announce,
    LiveRegion: () => <LiveRegion message={message} priority={priority} />
  };
}