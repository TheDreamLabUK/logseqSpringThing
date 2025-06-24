/**
 * Enhanced Toast Component
 * Modern toast notifications with animations
 */

import * as React from 'react'
import * as ToastPrimitive from '@radix-ui/react-toast'
import { motion, AnimatePresence } from 'framer-motion'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../utils/utils'
import { animations } from '../animations'

const toastVariants = cva(
  'relative flex items-start gap-3 w-full rounded-lg p-4 shadow-lg pointer-events-auto overflow-hidden',
  {
    variants: {
      variant: {
        default: 'bg-background border border-border',
        success: 'bg-success/10 border border-success/20 text-success-foreground',
        error: 'bg-destructive/10 border border-destructive/20 text-destructive-foreground',
        warning: 'bg-warning/10 border border-warning/20 text-warning-foreground',
        info: 'bg-primary/10 border border-primary/20 text-primary-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
)

const toastAnimationVariants = {
  initial: { opacity: 0, y: 50, scale: 0.9 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, x: 100, transition: { duration: 0.2 } },
}

interface ToastProps
  extends React.ComponentPropsWithoutRef<typeof ToastPrimitive.Root>,
    VariantProps<typeof toastVariants> {
  title?: string
  description?: string
  action?: React.ReactNode
  icon?: React.ReactNode
  closable?: boolean
  duration?: number
  onClose?: () => void
}

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitive.Root>,
  ToastProps
>(
  (
    {
      className,
      variant,
      title,
      description,
      action,
      icon,
      closable = true,
      duration = 5000,
      onClose,
      ...props
    },
    ref
  ) => {
    const [progress, setProgress] = React.useState(100)
    
    React.useEffect(() => {
      if (duration && duration !== Infinity) {
        const interval = setInterval(() => {
          setProgress((prev) => {
            if (prev <= 0) {
              clearInterval(interval)
              return 0
            }
            return prev - (100 / (duration / 100))
          })
        }, 100)
        
        return () => clearInterval(interval)
      }
    }, [duration])
    
    const defaultIcons = {
      success: (
        <svg className="h-5 w-5 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      error: (
        <svg className="h-5 w-5 text-destructive" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      warning: (
        <svg className="h-5 w-5 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
      info: (
        <svg className="h-5 w-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    }
    
    const displayIcon = icon || (variant && variant !== 'default' ? defaultIcons[variant] : null)
    
    return (
      <ToastPrimitive.Root
        ref={ref}
        duration={duration}
        className={cn(toastVariants({ variant }), className)}
        onOpenChange={(open) => {
          if (!open && onClose) {
            onClose()
          }
        }}
        {...props}
      >
        <motion.div
          className="flex items-start gap-3 flex-1"
          variants={toastAnimationVariants}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={animations.transitions.spring.smooth}
        >
          {displayIcon && (
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={animations.transitions.spring.bounce}
            >
              {displayIcon}
            </motion.div>
          )}
          
          <div className="flex-1">
            {title && (
              <ToastPrimitive.Title className="text-sm font-semibold">
                {title}
              </ToastPrimitive.Title>
            )}
            {description && (
              <ToastPrimitive.Description className="mt-1 text-sm opacity-90">
                {description}
              </ToastPrimitive.Description>
            )}
          </div>
          
          {action}
          
          {closable && (
            <ToastPrimitive.Close asChild>
              <motion.button
                className="ml-auto rounded-md p-1 hover:bg-foreground/10 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                transition={animations.transitions.spring.snappy}
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </motion.button>
            </ToastPrimitive.Close>
          )}
        </motion.div>
        
        {/* Progress bar */}
        {duration && duration !== Infinity && (
          <motion.div
            className="absolute bottom-0 left-0 h-1 bg-current opacity-20"
            initial={{ scaleX: 1 }}
            animate={{ scaleX: progress / 100 }}
            transition={{ duration: 0.1, ease: 'linear' }}
            style={{ originX: 0 }}
          />
        )}
      </ToastPrimitive.Root>
    )
  }
)
Toast.displayName = 'Toast'

// Toast Provider Component
interface ToastProviderProps extends ToastPrimitive.ToastProviderProps {
  children: React.ReactNode
  position?: 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right'
}

const ToastProvider = ({ children, position = 'bottom-right', ...props }: ToastProviderProps) => {
  const positionClasses = {
    'top-left': 'top-0 left-0',
    'top-center': 'top-0 left-1/2 -translate-x-1/2',
    'top-right': 'top-0 right-0',
    'bottom-left': 'bottom-0 left-0',
    'bottom-center': 'bottom-0 left-1/2 -translate-x-1/2',
    'bottom-right': 'bottom-0 right-0',
  }
  
  return (
    <ToastPrimitive.Provider {...props}>
      {children}
      <ToastPrimitive.Viewport
        className={cn(
          'fixed z-[100] flex max-h-screen w-full flex-col p-4 sm:max-w-[420px]',
          positionClasses[position]
        )}
      />
    </ToastPrimitive.Provider>
  )
}

// Toast Hook
interface ToastData {
  id: string
  title?: string
  description?: string
  variant?: 'default' | 'success' | 'error' | 'warning' | 'info'
  duration?: number
  action?: React.ReactNode
  icon?: React.ReactNode
}

interface ToastContextValue {
  toasts: ToastData[]
  toast: (data: Omit<ToastData, 'id'>) => void
  dismiss: (id: string) => void
  dismissAll: () => void
}

const ToastContext = React.createContext<ToastContextValue | undefined>(undefined)

export const useToast = () => {
  const context = React.useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

// Enhanced Toast Provider with Context
export const EnhancedToastProvider = ({ children, ...props }: ToastProviderProps) => {
  const [toasts, setToasts] = React.useState<ToastData[]>([])
  
  const toast = React.useCallback((data: Omit<ToastData, 'id'>) => {
    const id = Math.random().toString(36).substring(2, 9)
    setToasts((prev) => [...prev, { ...data, id }])
  }, [])
  
  const dismiss = React.useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])
  
  const dismissAll = React.useCallback(() => {
    setToasts([])
  }, [])
  
  return (
    <ToastContext.Provider value={{ toasts, toast, dismiss, dismissAll }}>
      <ToastProvider {...props}>
        {children}
        <AnimatePresence mode="popLayout">
          {toasts.map((toastData) => (
            <Toast
              key={toastData.id}
              {...toastData}
              onClose={() => dismiss(toastData.id)}
            />
          ))}
        </AnimatePresence>
      </ToastProvider>
    </ToastContext.Provider>
  )
}

// Pre-configured toast functions
export const toast = {
  success: (title: string, description?: string) => ({
    title,
    description,
    variant: 'success' as const,
  }),
  error: (title: string, description?: string) => ({
    title,
    description,
    variant: 'error' as const,
  }),
  warning: (title: string, description?: string) => ({
    title,
    description,
    variant: 'warning' as const,
  }),
  info: (title: string, description?: string) => ({
    title,
    description,
    variant: 'info' as const,
  }),
}

export { Toast, ToastProvider }