import React, { useEffect, useState } from 'react';
import { cn } from '../utils/cn';

/**
 * Global focus indicator that shows when keyboard navigation is active
 * Enhances visibility of focused elements for keyboard users
 */
export function FocusIndicator() {
  const [isKeyboardNav, setIsKeyboardNav] = useState(false);

  useEffect(() => {
    let lastMouseTime = 0;
    let keyboardThrottleTimer: NodeJS.Timeout | null = null;

    const handleMouseMove = () => {
      lastMouseTime = Date.now();
      if (isKeyboardNav) {
        setIsKeyboardNav(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if it's been at least 50ms since last mouse movement
      if (Date.now() - lastMouseTime > 50) {
        // Only set keyboard nav for navigation keys
        if (['Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter', ' '].includes(e.key)) {
          if (!isKeyboardNav) {
            setIsKeyboardNav(true);
          }
          
          // Clear any existing timer
          if (keyboardThrottleTimer) {
            clearTimeout(keyboardThrottleTimer);
          }
          
          // Set a timer to turn off keyboard nav after inactivity
          keyboardThrottleTimer = setTimeout(() => {
            setIsKeyboardNav(false);
          }, 5000);
        }
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('keydown', handleKeyDown);
      if (keyboardThrottleTimer) {
        clearTimeout(keyboardThrottleTimer);
      }
    };
  }, [isKeyboardNav]);

  // Add or remove keyboard navigation class to body
  useEffect(() => {
    if (isKeyboardNav) {
      document.body.classList.add('keyboard-nav');
    } else {
      document.body.classList.remove('keyboard-nav');
    }
  }, [isKeyboardNav]);

  return null; // This component doesn't render anything visible
}

/**
 * Enhanced focus styles to be added to global CSS
 */
export const focusIndicatorStyles = `
  /* Remove default focus outline when not using keyboard */
  *:focus {
    outline: none;
  }

  /* Show enhanced focus styles only during keyboard navigation */
  .keyboard-nav *:focus,
  .keyboard-nav *:focus-visible {
    outline: 2px solid var(--ring);
    outline-offset: 2px;
    border-radius: 4px;
  }

  /* Special handling for specific components */
  .keyboard-nav button:focus-visible,
  .keyboard-nav a:focus-visible,
  .keyboard-nav input:focus-visible,
  .keyboard-nav textarea:focus-visible,
  .keyboard-nav select:focus-visible {
    box-shadow: 0 0 0 2px var(--background), 0 0 0 4px var(--ring);
  }

  /* Skip link styles */
  .skip-link:focus {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 9999;
    padding: 1rem;
    background: var(--primary);
    color: var(--primary-foreground);
    text-decoration: none;
    border-radius: 0 0 0.5rem 0;
  }
`;