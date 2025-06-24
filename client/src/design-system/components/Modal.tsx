/**
 * Enhanced Modal Component
 * Modern modal with animations and advanced features
 */

import * as React from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { motion, AnimatePresence } from 'framer-motion'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../utils/utils'
import { animations } from '../animations'

const modalVariants = cva(
  'fixed z-50 w-full bg-background shadow-2xl',
  {
    variants: {
      size: {
        xs: 'max-w-xs',
        sm: 'max-w-sm',
        md: 'max-w-md',
        lg: 'max-w-lg',
        xl: 'max-w-xl',
        '2xl': 'max-w-2xl',
        '3xl': 'max-w-3xl',
        '4xl': 'max-w-4xl',
        '5xl': 'max-w-5xl',
        full: 'max-w-full h-full',
        auto: 'max-w-fit',
      },
      position: {
        center: 'left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] rounded-xl',
        top: 'left-[50%] top-0 -translate-x-[50%] rounded-b-xl',
        bottom: 'left-[50%] bottom-0 -translate-x-[50%] rounded-t-xl',
        left: 'left-0 top-[50%] -translate-y-[50%] h-full rounded-r-xl',
        right: 'right-0 top-[50%] -translate-y-[50%] h-full rounded-l-xl',
      },
    },
    defaultVariants: {
      size: 'md',
      position: 'center',
    },
  }
)

const overlayVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
}

const modalAnimationVariants = {
  center: {
    initial: { opacity: 0, scale: 0.9, y: 20 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exit: { opacity: 0, scale: 0.9, y: 20 },
  },
  top: {
    initial: { opacity: 0, y: -100 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -100 },
  },
  bottom: {
    initial: { opacity: 0, y: 100 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 100 },
  },
  left: {
    initial: { opacity: 0, x: -100 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -100 },
  },
  right: {
    initial: { opacity: 0, x: 100 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 100 },
  },
}

export interface ModalProps extends VariantProps<typeof modalVariants> {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  children: React.ReactNode
  trigger?: React.ReactNode
  closeOnOverlayClick?: boolean
  closeOnEscape?: boolean
  showCloseButton?: boolean
  overlayBlur?: boolean
  preventScroll?: boolean
}

const Modal = ({
  open,
  onOpenChange,
  children,
  trigger,
  size,
  position = 'center',
  closeOnOverlayClick = true,
  closeOnEscape = true,
  showCloseButton = true,
  overlayBlur = false,
  preventScroll = true,
}: ModalProps) => {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      {trigger && <DialogPrimitive.Trigger asChild>{trigger}</DialogPrimitive.Trigger>}
      <AnimatePresence>
        {open && (
          <DialogPrimitive.Portal forceMount>
            <DialogPrimitive.Overlay asChild>
              <motion.div
                className={cn(
                  'fixed inset-0 z-50 bg-black/50',
                  overlayBlur && 'backdrop-blur-sm',
                  preventScroll && 'overflow-hidden'
                )}
                variants={overlayVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={animations.transitions.easing.easeOut}
                onClick={closeOnOverlayClick ? () => onOpenChange?.(false) : undefined}
              />
            </DialogPrimitive.Overlay>
            <DialogPrimitive.Content
              asChild
              onEscapeKeyDown={closeOnEscape ? undefined : (e) => e.preventDefault()}
            >
              <motion.div
                className={cn(modalVariants({ size, position }))}
                variants={modalAnimationVariants[position]}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={animations.transitions.spring.smooth}
                onClick={(e) => e.stopPropagation()}
              >
                {showCloseButton && (
                  <DialogPrimitive.Close asChild>
                    <motion.button
                      className="absolute right-4 top-4 rounded-lg p-2 hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-primary/20"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      transition={animations.transitions.spring.snappy}
                    >
                      <svg
                        className="h-5 w-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                      <span className="sr-only">Close</span>
                    </motion.button>
                  </DialogPrimitive.Close>
                )}
                {children}
              </motion.div>
            </DialogPrimitive.Content>
          </DialogPrimitive.Portal>
        )}
      </AnimatePresence>
    </DialogPrimitive.Root>
  )
}

// Modal Header Component
interface ModalHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  bordered?: boolean
  sticky?: boolean
}

const ModalHeader = React.forwardRef<HTMLDivElement, ModalHeaderProps>(
  ({ className, bordered = false, sticky = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'flex flex-col space-y-1.5 p-6',
        bordered && 'border-b border-border',
        sticky && 'sticky top-0 z-10 bg-background',
        className
      )}
      {...props}
    />
  )
)
ModalHeader.displayName = 'ModalHeader'

// Modal Title Component
const ModalTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title asChild>
    <h2
      ref={ref}
      className={cn('text-xl font-semibold leading-tight tracking-tight', className)}
      {...props}
    />
  </DialogPrimitive.Title>
))
ModalTitle.displayName = 'ModalTitle'

// Modal Description Component
const ModalDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description asChild>
    <p ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
  </DialogPrimitive.Description>
))
ModalDescription.displayName = 'ModalDescription'

// Modal Body Component
interface ModalBodyProps extends React.HTMLAttributes<HTMLDivElement> {
  noPadding?: boolean
}

const ModalBody = React.forwardRef<HTMLDivElement, ModalBodyProps>(
  ({ className, noPadding = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        !noPadding && 'p-6',
        'flex-1 overflow-auto',
        className
      )}
      {...props}
    />
  )
)
ModalBody.displayName = 'ModalBody'

// Modal Footer Component
interface ModalFooterProps extends React.HTMLAttributes<HTMLDivElement> {
  bordered?: boolean
  sticky?: boolean
}

const ModalFooter = React.forwardRef<HTMLDivElement, ModalFooterProps>(
  ({ className, bordered = false, sticky = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'flex items-center justify-end gap-3 p-6',
        bordered && 'border-t border-border',
        sticky && 'sticky bottom-0 z-10 bg-background',
        className
      )}
      {...props}
    />
  )
)
ModalFooter.displayName = 'ModalFooter'

// Confirmation Modal Component
interface ConfirmationModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
  onCancel?: () => void
  title: string
  description?: string
  confirmText?: string
  cancelText?: string
  type?: 'default' | 'danger' | 'warning'
  loading?: boolean
}

const ConfirmationModal = ({
  open,
  onOpenChange,
  onConfirm,
  onCancel,
  title,
  description,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  type = 'default',
  loading = false,
}: ConfirmationModalProps) => {
  const handleCancel = () => {
    onCancel?.()
    onOpenChange(false)
  }
  
  const handleConfirm = () => {
    onConfirm()
    if (!loading) {
      onOpenChange(false)
    }
  }
  
  const iconVariants = {
    default: (
      <svg className="h-6 w-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    danger: (
      <svg className="h-6 w-6 text-destructive" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    warning: (
      <svg className="h-6 w-6 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  }
  
  return (
    <Modal open={open} onOpenChange={onOpenChange} size="sm">
      <ModalBody>
        <div className="flex items-start gap-4">
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={animations.transitions.spring.bounce}
          >
            {iconVariants[type]}
          </motion.div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold">{title}</h3>
            {description && (
              <p className="mt-2 text-sm text-muted-foreground">{description}</p>
            )}
          </div>
        </div>
      </ModalBody>
      <ModalFooter>
        <motion.button
          className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          onClick={handleCancel}
          disabled={loading}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {cancelText}
        </motion.button>
        <motion.button
          className={cn(
            'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
            type === 'danger'
              ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90'
              : type === 'warning'
              ? 'bg-warning text-warning-foreground hover:bg-warning/90'
              : 'bg-primary text-primary-foreground hover:bg-primary/90'
          )}
          onClick={handleConfirm}
          disabled={loading}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {loading ? 'Loading...' : confirmText}
        </motion.button>
      </ModalFooter>
    </Modal>
  )
}

export {
  Modal,
  ModalHeader,
  ModalTitle,
  ModalDescription,
  ModalBody,
  ModalFooter,
  ConfirmationModal,
}