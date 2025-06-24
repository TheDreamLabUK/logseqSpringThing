/**
 * Enhanced Button Component
 * Modern button with advanced animations and variants
 */

import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '../../utils/cn'
import { animations } from '../../lib/design-system/animations'

const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden',
  {
    variants: {
      variant: {
        default:
          'bg-primary text-primary-foreground shadow-md hover:shadow-lg active:shadow-sm',
        destructive:
          'bg-destructive text-destructive-foreground shadow-sm hover:shadow-md',
        outline:
          'border-2 border-input bg-transparent shadow-sm hover:bg-accent hover:text-accent-foreground',
        secondary:
          'bg-secondary text-secondary-foreground shadow-sm hover:shadow-md',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
        gradient:
          'bg-gradient-to-r from-primary to-primary/80 text-primary-foreground shadow-md hover:shadow-lg',
        glow:
          'bg-primary text-primary-foreground shadow-primary/25 shadow-lg hover:shadow-primary/40 hover:shadow-xl',
      },
      size: {
        xs: 'h-7 px-2 text-xs rounded-md',
        sm: 'h-8 px-3 text-sm rounded-md',
        default: 'h-10 px-4 py-2 text-base rounded-lg',
        lg: 'h-12 px-6 text-lg rounded-lg',
        xl: 'h-14 px-8 text-xl rounded-xl',
        icon: 'h-10 w-10 rounded-lg',
        'icon-sm': 'h-8 w-8 rounded-md',
        'icon-lg': 'h-12 w-12 rounded-lg',
      },
      fullWidth: {
        true: 'w-full',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  loading?: boolean
  loadingText?: string
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
  pulse?: boolean
  ripple?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      fullWidth,
      asChild = false,
      loading = false,
      loadingText = 'Loading...',
      disabled,
      children,
      icon,
      iconPosition = 'left',
      pulse = false,
      ripple = true,
      onClick,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : motion.button
    const isDisabled = disabled || loading
    const [ripples, setRipples] = React.useState<{ x: number; y: number; id: number }[]>([])

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (ripple && !isDisabled) {
        const rect = e.currentTarget.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top
        const id = Date.now()

        setRipples((prev) => [...prev, { x, y, id }])
        setTimeout(() => {
          setRipples((prev) => prev.filter((r) => r.id !== id))
        }, 600)
      }

      onClick?.(e)
    }

    const buttonContent = (
      <>
        {/* Ripple effect */}
        <AnimatePresence>
          {ripples.map((ripple) => (
            <motion.span
              key={ripple.id}
              className="absolute bg-white/30 rounded-full pointer-events-none"
              style={{
                left: ripple.x,
                top: ripple.y,
                width: 10,
                height: 10,
                x: '-50%',
                y: '-50%',
              }}
              initial={{ scale: 0, opacity: 1 }}
              animate={{ scale: 20, opacity: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
            />
          ))}
        </AnimatePresence>

        {/* Button content */}
        <span className="relative z-10 flex items-center gap-2">
          {loading ? (
            <>
              <motion.svg
                className="h-4 w-4"
                viewBox="0 0 24 24"
                variants={animations.loadingVariants.spinner}
                animate="animate"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </motion.svg>
              <span className="sr-only">{loadingText}</span>
              <span aria-hidden="true">{loadingText}</span>
            </>
          ) : (
            <>
              {icon && iconPosition === 'left' && icon}
              {children}
              {icon && iconPosition === 'right' && icon}
            </>
          )}
        </span>

        {/* Pulse animation */}
        {pulse && !isDisabled && (
          <motion.span
            className="absolute inset-0 rounded-inherit"
            style={{
              background: 'radial-gradient(circle, currentColor 0%, transparent 70%)',
              opacity: 0.4,
            }}
            variants={animations.loadingVariants.pulse}
            animate="animate"
          />
        )}
      </>
    )

    if (asChild) {
      return (
        <Slot
          className={cn(buttonVariants({ variant, size, fullWidth, className }))}
          ref={ref}
          aria-disabled={isDisabled}
          aria-busy={loading}
          onClick={handleClick}
        >
          {buttonContent}
        </Slot>
      )
    }

    return (
      <Comp
        className={cn(buttonVariants({ variant, size, fullWidth, className }))}
        ref={ref}
        disabled={isDisabled}
        aria-disabled={isDisabled}
        aria-busy={loading}
        onClick={handleClick}
        whileHover={!isDisabled ? animations.interactionVariants.hover : undefined}
        whileTap={!isDisabled ? animations.interactionVariants.tap : undefined}
        transition={animations.transitions.spring.smooth}
      >
        {buttonContent}
      </Comp>
    )
  }
)
Button.displayName = 'Button'

// Button Group Component
interface ButtonGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  orientation?: 'horizontal' | 'vertical'
  spacing?: 'none' | 'sm' | 'md' | 'lg'
}

const ButtonGroup = React.forwardRef<HTMLDivElement, ButtonGroupProps>(
  ({ className, orientation = 'horizontal', spacing = 'sm', children, ...props }, ref) => {
    const spacingClasses = {
      none: '',
      sm: orientation === 'horizontal' ? 'gap-2' : 'gap-2',
      md: orientation === 'horizontal' ? 'gap-4' : 'gap-4',
      lg: orientation === 'horizontal' ? 'gap-6' : 'gap-6',
    }

    return (
      <div
        ref={ref}
        className={cn(
          'flex',
          orientation === 'horizontal' ? 'flex-row' : 'flex-col',
          spacingClasses[spacing],
          className
        )}
        {...props}
      >
        {children}
      </div>
    )
  }
)
ButtonGroup.displayName = 'ButtonGroup'

export { Button, ButtonGroup, buttonVariants }