# Design System

A comprehensive design system built with React, Radix UI, Framer Motion, and Tailwind CSS.

## Overview

This design system provides a complete set of components, patterns, and utilities for building modern, accessible, and animated user interfaces.

### Key Features

- **Modern Components**: Enhanced Radix UI components with advanced animations
- **Design Tokens**: Comprehensive token system for consistent styling
- **Animation System**: Pre-built animations using Framer Motion
- **Theme Support**: Light/dark mode with smooth transitions
- **TypeScript**: Full type safety and IntelliSense support
- **Accessibility**: WCAG compliant components

## Components

### Core Components

#### Button
Enhanced button with multiple variants, sizes, and states.

```tsx
import { Button } from '@/design-system'

// Variants
<Button variant="default">Default</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="destructive">Destructive</Button>
<Button variant="gradient">Gradient</Button>
<Button variant="glow">Glow Effect</Button>

// With loading state
<Button loading loadingText="Processing...">Submit</Button>

// With icon
<Button icon={<IconComponent />}>Download</Button>
```

#### Card
Flexible card component with animation options.

```tsx
import { Card, AnimatedCard } from '@/design-system'

// Basic card
<Card variant="elevated">
  <CardHeader>
    <CardTitle>Title</CardTitle>
    <CardDescription>Description</CardDescription>
  </CardHeader>
  <CardContent>Content</CardContent>
</Card>

// Animated card
<AnimatedCard animationType="lift">
  <CardContent>Hover me!</CardContent>
</AnimatedCard>
```

#### Input
Advanced input with floating labels and validation states.

```tsx
import { Input, SearchInput } from '@/design-system'

// With floating label
<Input
  label="Email"
  type="email"
  floatingLabel
  error={errors.email}
/>

// Search input
<SearchInput
  placeholder="Search..."
  onSearch={handleSearch}
/>
```

#### Modal
Flexible modal with animations and positions.

```tsx
import { Modal, ConfirmationModal } from '@/design-system'

// Standard modal
<Modal open={open} onOpenChange={setOpen} size="lg">
  <ModalHeader>
    <ModalTitle>Title</ModalTitle>
  </ModalHeader>
  <ModalBody>Content</ModalBody>
  <ModalFooter>
    <Button>Action</Button>
  </ModalFooter>
</Modal>

// Confirmation modal
<ConfirmationModal
  open={open}
  onOpenChange={setOpen}
  onConfirm={handleConfirm}
  title="Are you sure?"
  type="danger"
/>
```

#### Toast
Toast notifications with variants.

```tsx
import { useToast, toast } from '@/design-system'

const { toast: showToast } = useToast()

// Show toasts
showToast(toast.success('Success!', 'Operation completed'))
showToast(toast.error('Error!', 'Something went wrong'))
showToast(toast.warning('Warning!', 'Please review'))
showToast(toast.info('Info', 'New update available'))
```

## Animation System

Pre-built animations for consistent motion design.

### Variants

```tsx
import { animations } from '@/design-system'

// Fade animation
<motion.div variants={animations.variants.fade}>
  Content
</motion.div>

// Slide animations
<motion.div variants={animations.variants.slideUp}>
  Slides up
</motion.div>

// Page transitions
<motion.div variants={animations.pageTransitions.fadeScale}>
  Page content
</motion.div>
```

### Stagger Children

```tsx
<motion.div variants={animations.staggerVariants.container}>
  {items.map(item => (
    <motion.div
      key={item.id}
      variants={animations.staggerVariants.item}
    >
      {item.content}
    </motion.div>
  ))}
</motion.div>
```

## Theme System

### Setup

```tsx
import { ThemeProvider } from '@/design-system'

function App() {
  return (
    <ThemeProvider defaultTheme="system">
      <YourApp />
    </ThemeProvider>
  )
}
```

### Using Theme

```tsx
import { useTheme, ThemeToggle } from '@/design-system'

function Component() {
  const { theme, setTheme, toggleTheme } = useTheme()
  
  return (
    <>
      <ThemeToggle />
      <button onClick={toggleTheme}>
        Toggle Theme
      </button>
    </>
  )
}
```

## Design Tokens

Access design tokens programmatically:

```tsx
import { tokens } from '@/design-system'

// Colors
tokens.colors.primary[500]
tokens.colors.success[400]

// Typography
tokens.typography.fontSize.lg
tokens.typography.fontWeight.semibold

// Spacing
tokens.spacing[4]  // 1rem
tokens.spacing[8]  // 2rem

// Animations
tokens.animation.duration.normal  // 250ms
tokens.animation.easing.easeInOut
```

## Patterns

### Form Patterns

```tsx
import { LoginFormPattern, MultiStepFormPattern } from '@/design-system/patterns'

// Login form with validation
<LoginFormPattern />

// Multi-step form with progress
<MultiStepFormPattern />
```

## Best Practices

1. **Consistency**: Use design tokens instead of hard-coded values
2. **Accessibility**: All components follow WCAG guidelines
3. **Performance**: Components are optimized with React.memo where appropriate
4. **Animation**: Use the animation system for consistent motion
5. **Theming**: Support both light and dark modes

## Migration Guide

To migrate existing components:

1. Replace UI imports with design system imports
2. Update component props to match new APIs
3. Apply animation variants where appropriate
4. Use design tokens for custom styles

```tsx
// Before
import { Button } from '@/ui/Button'
<Button className="bg-blue-500">Click</Button>

// After
import { Button } from '@/design-system'
<Button variant="primary">Click</Button>
```