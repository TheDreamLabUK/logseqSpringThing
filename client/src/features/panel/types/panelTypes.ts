import React from 'react';

// Detachable section props
export interface DetachableSectionProps {
  children: React.ReactNode;
  title: string;
  defaultDetached?: boolean;
  defaultPosition?: { x: number; y: number };
}