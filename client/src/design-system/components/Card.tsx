/**
 * Enhanced Card Component
 * Modern card with animations and interactive features
 */

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../../utils/utils'
import { animations } from '../animations'

const cardVariants = cva(
  'rounded-xl bg-card text-card-foreground transition-all duration-200',
  {
    variants: {
      variant: {
        default: 'border border-border shadow-sm',
        elevated: 'shadow-md hover:shadow-lg',
        outlined: 'border-2 border-border',
        ghost: 'hover:bg-accent/5',
        gradient: 'bg-gradient-to-br from-card to-card/80 border border-border/50',
        glass: 'backdrop-blur-md bg-card/80 border border-border/50',
      },
      padding: {
        none: '',
        sm: 'p-4',
        md: 'p-6',
        lg: 'p-8',
        xl: 'p-10',
      },
      interactive: {
        true: 'cursor-pointer',
      },
    },
    defaultVariants: {
      variant: 'default',
      padding: 'md',
    },
  }
)

export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {
  as?: 'div' | 'article' | 'section'
  hoverable?: boolean
  pressable?: boolean
  animated?: boolean
  blurred?: boolean
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      className,
      variant,
      padding,
      interactive,
      as: Component = 'div',
      hoverable = false,
      pressable = false,
      animated = true,
      blurred = false,
      children,
      ...props
    },
    ref
  ) => {
    const MotionComponent = motion[Component as keyof typeof motion] as any
    
    const cardContent = (
      <Component
        ref={ref}
        className={cn(
          cardVariants({ variant, padding, interactive: interactive || hoverable || pressable }),
          blurred && 'backdrop-blur-sm',
          className
        )}
        {...props}
      >
        {children}
      </Component>
    )
    
    if (!animated) {
      return cardContent
    }
    
    return (
      <MotionComponent
        ref={ref}
        className={cn(
          cardVariants({ variant, padding, interactive: interactive || hoverable || pressable }),
          blurred && 'backdrop-blur-sm',
          className
        )}
        initial={false}
        whileHover={
          hoverable
            ? {
                scale: 1.02,
                transition: animations.transitions.spring.smooth,
              }
            : undefined
        }
        whileTap={
          pressable
            ? {
                scale: 0.98,
                transition: animations.transitions.spring.snappy,
              }
            : undefined
        }
        {...props}
      >
        {children}
      </MotionComponent>
    )
  }
)
Card.displayName = 'Card'

// Card Header Component
interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  bordered?: boolean
  sticky?: boolean
}

const CardHeader = React.forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className, bordered = false, sticky = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'flex flex-col space-y-1.5',
        bordered && 'border-b border-border pb-6',
        sticky && 'sticky top-0 z-10 bg-card',
        className
      )}
      {...props}
    />
  )
)
CardHeader.displayName = 'CardHeader'

// Card Title Component
const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn('text-xl font-semibold leading-tight tracking-tight', className)}
    {...props}
  />
))
CardTitle.displayName = 'CardTitle'

// Card Description Component
const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
))
CardDescription.displayName = 'CardDescription'

// Card Content Component
interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {
  noPadding?: boolean
}

const CardContent = React.forwardRef<HTMLDivElement, CardContentProps>(
  ({ className, noPadding = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(!noPadding && 'px-6 pb-6', className)}
      {...props}
    />
  )
)
CardContent.displayName = 'CardContent'

// Card Footer Component
interface CardFooterProps extends React.HTMLAttributes<HTMLDivElement> {
  bordered?: boolean
  sticky?: boolean
}

const CardFooter = React.forwardRef<HTMLDivElement, CardFooterProps>(
  ({ className, bordered = false, sticky = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'flex items-center',
        bordered && 'border-t border-border pt-6',
        sticky && 'sticky bottom-0 z-10 bg-card',
        className
      )}
      {...props}
    />
  )
)
CardFooter.displayName = 'CardFooter'

// Card Stack Component for stacked cards
interface CardStackProps extends React.HTMLAttributes<HTMLDivElement> {
  spacing?: 'sm' | 'md' | 'lg'
  direction?: 'vertical' | 'horizontal'
}

const CardStack = React.forwardRef<HTMLDivElement, CardStackProps>(
  ({ className, spacing = 'md', direction = 'vertical', children, ...props }, ref) => {
    const spacingClasses = {
      sm: direction === 'vertical' ? 'space-y-2' : 'space-x-2',
      md: direction === 'vertical' ? 'space-y-4' : 'space-x-4',
      lg: direction === 'vertical' ? 'space-y-6' : 'space-x-6',
    }
    
    return (
      <motion.div
        ref={ref}
        className={cn(
          'flex',
          direction === 'vertical' ? 'flex-col' : 'flex-row',
          spacingClasses[spacing],
          className
        )}
        variants={animations.staggerVariants.container}
        initial="initial"
        animate="animate"
        {...props}
      >
        {React.Children.map(children, (child, index) => (
          <motion.div
            key={index}
            variants={animations.staggerVariants.item}
            custom={index}
          >
            {child}
          </motion.div>
        ))}
      </motion.div>
    )
  }
)
CardStack.displayName = 'CardStack'

// Animated Card for special interactions
interface AnimatedCardProps extends CardProps {
  animationType?: 'lift' | 'glow' | 'tilt' | 'flip'
}

const AnimatedCard = React.forwardRef<HTMLDivElement, AnimatedCardProps>(
  ({ animationType = 'lift', children, ...props }, ref) => {
    const [isFlipped, setIsFlipped] = React.useState(false)
    
    const animationVariants = {
      lift: {
        rest: { y: 0, boxShadow: 'var(--shadow-sm)' },
        hover: { y: -8, boxShadow: 'var(--shadow-lg)' },
      },
      glow: {
        rest: { boxShadow: 'var(--shadow-sm)' },
        hover: {
          boxShadow: '0 0 20px rgba(var(--color-primary), 0.3), var(--shadow-lg)',
        },
      },
      tilt: {
        rest: { rotateX: 0, rotateY: 0 },
        hover: { rotateX: -5, rotateY: 5 },
      },
      flip: {
        rest: { rotateY: 0 },
        hover: { rotateY: 180 },
      },
    }
    
    return (
      <motion.div
        ref={ref}
        initial="rest"
        whileHover="hover"
        animate={isFlipped && animationType === 'flip' ? 'hover' : 'rest'}
        variants={animationVariants[animationType]}
        transition={animations.transitions.spring.smooth}
        onClick={() => animationType === 'flip' && setIsFlipped(!isFlipped)}
        style={{ transformStyle: 'preserve-3d', perspective: 1000 }}
      >
        <Card {...props}>{children}</Card>
      </motion.div>
    )
  }
)
AnimatedCard.displayName = 'AnimatedCard'

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
  CardStack,
  AnimatedCard,
}