/**
 * Animation System
 * Framer Motion animation presets and utilities
 */

import { Variants, Transition } from 'framer-motion'
import { tokens } from '../../design-system/tokens'

// Animation transitions
export const transitions = {
  // Spring transitions
  spring: {
    bounce: {
      type: 'spring',
      damping: 15,
      stiffness: 300,
    },
    smooth: {
      type: 'spring',
      damping: 20,
      stiffness: 260,
    },
    snappy: {
      type: 'spring',
      damping: 25,
      stiffness: 400,
    },
    slow: {
      type: 'spring',
      damping: 30,
      stiffness: 150,
    },
  },

  // Easing transitions
  easing: {
    easeIn: {
      duration: parseFloat(tokens.animation.duration.normal) / 1000,
      ease: [0.4, 0, 1, 1],
    },
    easeOut: {
      duration: parseFloat(tokens.animation.duration.normal) / 1000,
      ease: [0, 0, 0.2, 1],
    },
    easeInOut: {
      duration: parseFloat(tokens.animation.duration.normal) / 1000,
      ease: [0.4, 0, 0.2, 1],
    },
  },
}

// Common animation variants
export const variants = {
  // Fade animations
  fade: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
  } as Variants,

  // Scale animations
  scale: {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.9 },
  } as Variants,

  // Slide animations
  slideUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  } as Variants,

  slideDown: {
    initial: { opacity: 0, y: -20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 20 },
  } as Variants,

  slideLeft: {
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -20 },
  } as Variants,

  slideRight: {
    initial: { opacity: 0, x: -20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 },
  } as Variants,

  // Expand animations
  expandVertical: {
    initial: { height: 0, opacity: 0 },
    animate: { height: 'auto', opacity: 1 },
    exit: { height: 0, opacity: 0 },
  } as Variants,

  expandHorizontal: {
    initial: { width: 0, opacity: 0 },
    animate: { width: 'auto', opacity: 1 },
    exit: { width: 0, opacity: 0 },
  } as Variants,

  // Rotate animations
  rotate: {
    initial: { opacity: 0, rotate: -10 },
    animate: { opacity: 1, rotate: 0 },
    exit: { opacity: 0, rotate: 10 },
  } as Variants,

  // Blur animations
  blur: {
    initial: { opacity: 0, filter: 'blur(10px)' },
    animate: { opacity: 1, filter: 'blur(0px)' },
    exit: { opacity: 0, filter: 'blur(10px)' },
  } as Variants,
}

// Stagger children animations
export const staggerVariants = {
  container: {
    animate: {
      transition: {
        staggerChildren: 0.1,
      },
    },
  } as Variants,

  containerFast: {
    animate: {
      transition: {
        staggerChildren: 0.05,
      },
    },
  } as Variants,

  containerSlow: {
    animate: {
      transition: {
        staggerChildren: 0.15,
      },
    },
  } as Variants,

  item: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
  } as Variants,
}

// Hover and tap animations
export const interactionVariants = {
  hover: {
    scale: 1.05,
    transition: transitions.spring.smooth,
  },

  tap: {
    scale: 0.95,
    transition: transitions.spring.snappy,
  },

  hoverLift: {
    y: -4,
    transition: transitions.spring.smooth,
  },

  hoverGlow: {
    boxShadow: tokens.shadows.lg,
    transition: transitions.easing.easeOut,
  },
}

// Page transition variants
export const pageTransitions = {
  fadeScale: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 1.05 },
    transition: transitions.easing.easeInOut,
  } as Variants,

  slideLeft: {
    initial: { x: '100%', opacity: 0 },
    animate: { x: 0, opacity: 1 },
    exit: { x: '-100%', opacity: 0 },
    transition: transitions.spring.smooth,
  } as Variants,

  slideUp: {
    initial: { y: '100%', opacity: 0 },
    animate: { y: 0, opacity: 1 },
    exit: { y: '-100%', opacity: 0 },
    transition: transitions.spring.smooth,
  } as Variants,
}

// Gesture animations
export const gestureAnimations = {
  drag: {
    dragElastic: 0.2,
    dragConstraints: { top: 0, left: 0, right: 0, bottom: 0 },
    whileDrag: { scale: 1.1, cursor: 'grabbing' },
  },

  pinch: {
    whileTap: { scale: 0.9 },
    whileInView: { scale: 1 },
  },
}

// Loading animations
export const loadingVariants = {
  spinner: {
    animate: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: 'linear',
      },
    },
  } as Variants,

  pulse: {
    animate: {
      scale: [1, 1.2, 1],
      opacity: [1, 0.5, 1],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  } as Variants,

  dots: {
    animate: {
      y: [0, -10, 0],
      transition: {
        duration: 0.6,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  } as Variants,
}

// Animation utility functions
export const animationUtils = {
  // Delay animation by index for stagger effect
  getStaggerDelay: (index: number, baseDelay = 0.1) => ({
    transition: {
      delay: index * baseDelay,
    },
  }),

  // Create custom spring animation
  createSpring: (stiffness = 260, damping = 20) => ({
    type: 'spring',
    stiffness,
    damping,
  }),

  // Create custom bezier curve animation
  createBezier: (curve: [number, number, number, number], duration = 0.3) => ({
    duration,
    ease: curve,
  }),
}

// Export all animations as a single object for convenience
export const animations = {
  transitions,
  variants,
  staggerVariants,
  interactionVariants,
  pageTransitions,
  gestureAnimations,
  loadingVariants,
  utils: animationUtils,
}