/**
 * Enhanced Input Component
 * Modern input with animations and advanced features
 */

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../utils/cn'
import { animations } from '../animations'

const inputVariants = cva(
  'flex w-full rounded-lg bg-background text-foreground transition-all duration-200 file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'border border-input focus:border-primary focus:ring-2 focus:ring-primary/20',
        filled: 'bg-secondary border border-transparent focus:border-primary focus:bg-background',
        flushed: 'border-0 border-b-2 border-input rounded-none px-0 focus:border-primary',
        ghost: 'border-0 focus:bg-accent/5',
        outlined: 'border-2 border-input focus:border-primary',
      },
      size: {
        xs: 'h-7 px-2 text-xs',
        sm: 'h-8 px-3 text-sm',
        default: 'h-10 px-4 text-base',
        lg: 'h-12 px-5 text-lg',
        xl: 'h-14 px-6 text-xl',
      },
      state: {
        default: '',
        error: 'border-destructive focus:border-destructive focus:ring-destructive/20',
        success: 'border-success focus:border-success focus:ring-success/20',
        warning: 'border-warning focus:border-warning focus:ring-warning/20',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
      state: 'default',
    },
  }
)

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'>,
    VariantProps<typeof inputVariants> {
  label?: string
  error?: string
  success?: string
  warning?: string
  helper?: string
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
  clearable?: boolean
  onClear?: () => void
  animated?: boolean
  floatingLabel?: boolean
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      type = 'text',
      variant,
      size,
      state: stateProp,
      label,
      error,
      success,
      warning,
      helper,
      icon,
      iconPosition = 'left',
      clearable = false,
      onClear,
      animated = true,
      floatingLabel = false,
      value,
      onChange,
      onFocus,
      onBlur,
      disabled,
      ...props
    },
    ref
  ) => {
    const [isFocused, setIsFocused] = React.useState(false)
    const [hasValue, setHasValue] = React.useState(!!value)

    // Determine state based on props
    const state = error ? 'error' : success ? 'success' : warning ? 'warning' : stateProp || 'default'
    const message = error || success || warning || helper

    React.useEffect(() => {
      setHasValue(!!value)
    }, [value])

    const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(true)
      onFocus?.(e)
    }

    const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(false)
      onBlur?.(e)
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setHasValue(!!e.target.value)
      onChange?.(e)
    }

    const handleClear = () => {
      if (onClear) {
        onClear()
      }
      setHasValue(false)
    }

    const inputElement = (
      <div className="relative w-full">
        {/* Floating Label */}
        {floatingLabel && label && (
          <motion.label
            className={cn(
              'absolute left-4 text-muted-foreground transition-all duration-200 pointer-events-none',
              'origin-left',
              size === 'xs' && 'text-xs',
              size === 'sm' && 'text-sm',
              size === 'lg' && 'text-lg',
              size === 'xl' && 'text-xl'
            )}
            initial={false}
            animate={{
              y: hasValue || isFocused ? -20 : 0,
              scale: hasValue || isFocused ? 0.85 : 1,
              color: isFocused ? 'var(--color-primary)' : 'var(--color-muted-foreground)',
            }}
            transition={animations.transitions.spring.smooth}
          >
            {label}
          </motion.label>
        )}

        {/* Input Container */}
        <div className="relative flex items-center">
          {/* Left Icon */}
          {icon && iconPosition === 'left' && (
            <div className="absolute left-3 text-muted-foreground">
              {icon}
            </div>
          )}

          {/* Input Field */}
          <input
            ref={ref}
            type={type}
            className={cn(
              inputVariants({ variant, size, state }),
              icon && iconPosition === 'left' && 'pl-10',
              icon && iconPosition === 'right' && 'pr-10',
              clearable && 'pr-10',
              floatingLabel && 'pt-4',
              className
            )}
            value={value}
            onChange={handleChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            disabled={disabled}
            aria-invalid={state === 'error'}
            aria-describedby={message ? `${props.id}-message` : undefined}
            {...props}
          />

          {/* Right Icon / Clear Button */}
          <div className="absolute right-3 flex items-center gap-2">
            {clearable && hasValue && !disabled && (
              <motion.button
                type="button"
                onClick={handleClear}
                className="text-muted-foreground hover:text-foreground transition-colors"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={animations.transitions.spring.snappy}
              >
                <svg
                  className="h-4 w-4"
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
              </motion.button>
            )}
            {icon && iconPosition === 'right' && (
              <div className="text-muted-foreground">{icon}</div>
            )}
          </div>
        </div>

        {/* Focus Line Animation */}
        {animated && variant === 'flushed' && (
          <motion.div
            className="absolute bottom-0 left-0 h-0.5 bg-primary"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: isFocused ? 1 : 0 }}
            transition={animations.transitions.spring.smooth}
            style={{ originX: 0.5 }}
          />
        )}
      </div>
    )

    return (
      <div className="w-full">
        {/* Standard Label */}
        {label && !floatingLabel && (
          <label
            htmlFor={props.id}
            className={cn(
              'block text-sm font-medium mb-1.5',
              state === 'error' && 'text-destructive',
              state === 'success' && 'text-success',
              state === 'warning' && 'text-warning'
            )}
          >
            {label}
          </label>
        )}

        {/* Input */}
        {inputElement}

        {/* Helper/Error Message */}
        <AnimatePresence mode="wait">
          {message && (
            <motion.p
              id={`${props.id}-message`}
              className={cn(
                'mt-1.5 text-sm',
                state === 'error' && 'text-destructive',
                state === 'success' && 'text-success',
                state === 'warning' && 'text-warning',
                state === 'default' && 'text-muted-foreground'
              )}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={animations.transitions.spring.snappy}
            >
              {message}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    )
  }
)
Input.displayName = 'Input'

// Input Group Component
interface InputGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  orientation?: 'horizontal' | 'vertical'
  spacing?: 'sm' | 'md' | 'lg'
}

const InputGroup = React.forwardRef<HTMLDivElement, InputGroupProps>(
  ({ className, orientation = 'vertical', spacing = 'md', ...props }, ref) => {
    const spacingClasses = {
      sm: orientation === 'horizontal' ? 'gap-2' : 'gap-2',
      md: orientation === 'horizontal' ? 'gap-4' : 'gap-4',
      lg: orientation === 'horizontal' ? 'gap-6' : 'gap-6',
    }

    return (
      <div
        ref={ref}
        className={cn(
          'flex',
          orientation === 'horizontal' ? 'flex-row items-end' : 'flex-col',
          spacingClasses[spacing],
          className
        )}
        {...props}
      />
    )
  }
)
InputGroup.displayName = 'InputGroup'

// Search Input Component
interface SearchInputProps extends Omit<InputProps, 'type' | 'icon' | 'iconPosition'> {
  onSearch?: (value: string) => void
}

const SearchInput = React.forwardRef<HTMLInputElement, SearchInputProps>(
  ({ onSearch, onChange, onKeyDown, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange?.(e)
      onSearch?.(e.target.value)
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onSearch?.(e.currentTarget.value)
      }
      onKeyDown?.(e)
    }

    return (
      <Input
        ref={ref}
        type="search"
        icon={
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        }
        iconPosition="left"
        clearable
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        {...props}
      />
    )
  }
)
SearchInput.displayName = 'SearchInput'

export { Input, InputGroup, SearchInput, inputVariants }