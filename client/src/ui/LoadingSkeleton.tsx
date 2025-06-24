import React from 'react';
import { cn } from '../utils/cn';

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-md bg-muted",
        className
      )}
    />
  );
}

interface SkeletonTextProps {
  lines?: number;
  className?: string;
  lineClassName?: string;
}

export function SkeletonText({ 
  lines = 3, 
  className,
  lineClassName 
}: SkeletonTextProps) {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn(
            "h-4",
            i === lines - 1 && "w-4/5",
            lineClassName
          )}
        />
      ))}
    </div>
  );
}

interface SkeletonCardProps {
  className?: string;
  showHeader?: boolean;
  showFooter?: boolean;
}

export function SkeletonCard({ 
  className,
  showHeader = true,
  showFooter = false
}: SkeletonCardProps) {
  return (
    <div className={cn("rounded-lg border bg-card p-6", className)}>
      {showHeader && (
        <div className="space-y-2 mb-4">
          <Skeleton className="h-6 w-1/3" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}
      
      <SkeletonText lines={3} />
      
      {showFooter && (
        <div className="mt-4 flex gap-2">
          <Skeleton className="h-9 w-20" />
          <Skeleton className="h-9 w-20" />
        </div>
      )}
    </div>
  );
}

interface SkeletonListProps {
  items?: number;
  className?: string;
  itemClassName?: string;
}

export function SkeletonList({ 
  items = 5,
  className,
  itemClassName
}: SkeletonListProps) {
  return (
    <div className={cn("space-y-3", className)}>
      {Array.from({ length: items }).map((_, i) => (
        <div
          key={i}
          className={cn(
            "flex items-center space-x-3 p-3",
            itemClassName
          )}
        >
          <Skeleton className="h-10 w-10 rounded-full" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-1/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
}

interface SkeletonSettingProps {
  className?: string;
}

export function SkeletonSetting({ className }: SkeletonSettingProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <div className="space-y-1 flex-1">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-48" />
        </div>
        <Skeleton className="h-9 w-20" />
      </div>
    </div>
  );
}