import React from 'react';
import { cn } from '../utils/cn';

interface SkipLink {
  id: string;
  label: string;
}

interface SkipNavigationProps {
  links?: SkipLink[];
}

/**
 * Skip navigation links for keyboard users
 * Provides quick access to main content areas
 */
export function SkipNavigation({ 
  links = [
    { id: 'main-content', label: 'Skip to main content' },
    { id: 'navigation', label: 'Skip to navigation' },
    { id: 'search', label: 'Skip to search' }
  ] 
}: SkipNavigationProps) {
  return (
    <div className="sr-only focus-within:not-sr-only">
      <div className="absolute top-0 left-0 z-[100] bg-background p-2">
        {links.map((link) => (
          <a
            key={link.id}
            href={`#${link.id}`}
            className={cn(
              "block px-4 py-2 mb-1 text-sm font-medium",
              "bg-primary text-primary-foreground rounded",
              "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
              "hover:bg-primary/90"
            )}
            onClick={(e) => {
              e.preventDefault();
              const element = document.getElementById(link.id);
              if (element) {
                element.scrollIntoView({ behavior: 'smooth' });
                element.focus({ preventScroll: true });
              }
            }}
          >
            {link.label}
          </a>
        ))}
      </div>
    </div>
  );
}