/**
 * Design System Tokens
 * Enhanced token system with semantic naming and comprehensive coverage
 */

export const tokens = {
  // Color Palette - Base colors
  colors: {
    // Primary brand colors
    primary: {
      50: 'hsl(210, 40%, 98%)',
      100: 'hsl(210, 40%, 96%)',
      200: 'hsl(210, 40%, 92%)',
      300: 'hsl(210, 40%, 86%)',
      400: 'hsl(210, 40%, 76%)',
      500: 'hsl(210, 40%, 64%)',
      600: 'hsl(210, 40%, 52%)',
      700: 'hsl(210, 40%, 40%)',
      800: 'hsl(210, 40%, 28%)',
      900: 'hsl(210, 40%, 16%)',
      950: 'hsl(210, 40%, 8%)',
    },
    
    // Semantic colors
    success: {
      50: 'hsl(140, 60%, 98%)',
      100: 'hsl(140, 60%, 96%)',
      200: 'hsl(140, 60%, 88%)',
      300: 'hsl(140, 60%, 76%)',
      400: 'hsl(140, 60%, 60%)',
      500: 'hsl(140, 60%, 48%)',
      600: 'hsl(140, 60%, 36%)',
      700: 'hsl(140, 60%, 24%)',
      800: 'hsl(140, 60%, 16%)',
      900: 'hsl(140, 60%, 8%)',
    },
    
    warning: {
      50: 'hsl(48, 96%, 98%)',
      100: 'hsl(48, 96%, 96%)',
      200: 'hsl(48, 96%, 88%)',
      300: 'hsl(48, 96%, 76%)',
      400: 'hsl(48, 96%, 60%)',
      500: 'hsl(48, 96%, 48%)',
      600: 'hsl(48, 96%, 36%)',
      700: 'hsl(48, 96%, 24%)',
      800: 'hsl(48, 96%, 16%)',
      900: 'hsl(48, 96%, 8%)',
    },
    
    error: {
      50: 'hsl(0, 84%, 98%)',
      100: 'hsl(0, 84%, 96%)',
      200: 'hsl(0, 84%, 88%)',
      300: 'hsl(0, 84%, 76%)',
      400: 'hsl(0, 84%, 60%)',
      500: 'hsl(0, 84%, 48%)',
      600: 'hsl(0, 84%, 36%)',
      700: 'hsl(0, 84%, 24%)',
      800: 'hsl(0, 84%, 16%)',
      900: 'hsl(0, 84%, 8%)',
    },
    
    neutral: {
      50: 'hsl(0, 0%, 98%)',
      100: 'hsl(0, 0%, 96%)',
      200: 'hsl(0, 0%, 90%)',
      300: 'hsl(0, 0%, 83%)',
      400: 'hsl(0, 0%, 64%)',
      500: 'hsl(0, 0%, 45%)',
      600: 'hsl(0, 0%, 32%)',
      700: 'hsl(0, 0%, 25%)',
      800: 'hsl(0, 0%, 15%)',
      900: 'hsl(0, 0%, 9%)',
      950: 'hsl(0, 0%, 4%)',
    },
  },
  
  // Typography
  typography: {
    fontFamily: {
      sans: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif',
      mono: 'JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
      display: 'Sora, Inter, ui-sans-serif, system-ui',
    },
    
    fontSize: {
      '2xs': '0.625rem',   // 10px
      xs: '0.75rem',       // 12px
      sm: '0.875rem',      // 14px
      base: '1rem',        // 16px
      lg: '1.125rem',      // 18px
      xl: '1.25rem',       // 20px
      '2xl': '1.5rem',     // 24px
      '3xl': '1.875rem',   // 30px
      '4xl': '2.25rem',    // 36px
      '5xl': '3rem',       // 48px
      '6xl': '3.75rem',    // 60px
      '7xl': '4.5rem',     // 72px
    },
    
    fontWeight: {
      thin: 100,
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
      extrabold: 800,
      black: 900,
    },
    
    lineHeight: {
      none: 1,
      tight: 1.25,
      snug: 1.375,
      normal: 1.5,
      relaxed: 1.625,
      loose: 1.75,
      body: 1.625,
      heading: 1.2,
    },
    
    letterSpacing: {
      tighter: '-0.05em',
      tight: '-0.025em',
      normal: '0',
      wide: '0.025em',
      wider: '0.05em',
      widest: '0.1em',
    },
  },
  
  // Spacing & Layout
  spacing: {
    0: '0',
    1: '0.25rem',    // 4px
    2: '0.5rem',     // 8px
    3: '0.75rem',    // 12px
    4: '1rem',       // 16px
    5: '1.25rem',    // 20px
    6: '1.5rem',     // 24px
    7: '1.75rem',    // 28px
    8: '2rem',       // 32px
    9: '2.25rem',    // 36px
    10: '2.5rem',    // 40px
    11: '2.75rem',   // 44px
    12: '3rem',      // 48px
    14: '3.5rem',    // 56px
    16: '4rem',      // 64px
    20: '5rem',      // 80px
    24: '6rem',      // 96px
    28: '7rem',      // 112px
    32: '8rem',      // 128px
    36: '9rem',      // 144px
    40: '10rem',     // 160px
    44: '11rem',     // 176px
    48: '12rem',     // 192px
    52: '13rem',     // 208px
    56: '14rem',     // 224px
    60: '15rem',     // 240px
    64: '16rem',     // 256px
    72: '18rem',     // 288px
    80: '20rem',     // 320px
    96: '24rem',     // 384px
  },
  
  // Border Radius
  borderRadius: {
    none: '0',
    sm: '0.125rem',   // 2px
    base: '0.25rem',  // 4px
    md: '0.375rem',   // 6px
    lg: '0.5rem',     // 8px
    xl: '0.75rem',    // 12px
    '2xl': '1rem',    // 16px
    '3xl': '1.5rem',  // 24px
    full: '9999px',
  },
  
  // Shadows
  shadows: {
    xs: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    sm: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.08)',
    base: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.08)',
    md: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.08)',
    lg: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.08)',
    xl: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    '2xl': '0 35px 60px -15px rgba(0, 0, 0, 0.3)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
    none: '0 0 #0000',
    
    // Colored shadows
    primary: '0 4px 6px -1px rgba(79, 70, 229, 0.1), 0 2px 4px -2px rgba(79, 70, 229, 0.08)',
    success: '0 4px 6px -1px rgba(34, 197, 94, 0.1), 0 2px 4px -2px rgba(34, 197, 94, 0.08)',
    warning: '0 4px 6px -1px rgba(251, 191, 36, 0.1), 0 2px 4px -2px rgba(251, 191, 36, 0.08)',
    error: '0 4px 6px -1px rgba(239, 68, 68, 0.1), 0 2px 4px -2px rgba(239, 68, 68, 0.08)',
  },
  
  // Animation
  animation: {
    duration: {
      instant: '50ms',
      fast: '150ms',
      normal: '250ms',
      slow: '350ms',
      slower: '500ms',
      slowest: '700ms',
    },
    
    easing: {
      linear: 'linear',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeInSine: 'cubic-bezier(0.12, 0, 0.39, 0)',
      easeOutSine: 'cubic-bezier(0.61, 1, 0.88, 1)',
      easeInOutSine: 'cubic-bezier(0.37, 0, 0.63, 1)',
      easeInQuad: 'cubic-bezier(0.11, 0, 0.5, 0)',
      easeOutQuad: 'cubic-bezier(0.5, 1, 0.89, 1)',
      easeInOutQuad: 'cubic-bezier(0.45, 0, 0.55, 1)',
      easeInCubic: 'cubic-bezier(0.32, 0, 0.67, 0)',
      easeOutCubic: 'cubic-bezier(0.33, 1, 0.68, 1)',
      easeInOutCubic: 'cubic-bezier(0.65, 0, 0.35, 1)',
      easeInQuart: 'cubic-bezier(0.5, 0, 0.75, 0)',
      easeOutQuart: 'cubic-bezier(0.25, 1, 0.5, 1)',
      easeInOutQuart: 'cubic-bezier(0.76, 0, 0.24, 1)',
      easeInExpo: 'cubic-bezier(0.7, 0, 0.84, 0)',
      easeOutExpo: 'cubic-bezier(0.16, 1, 0.3, 1)',
      easeInOutExpo: 'cubic-bezier(0.87, 0, 0.13, 1)',
      easeInBack: 'cubic-bezier(0.36, 0, 0.66, -0.56)',
      easeOutBack: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
      easeInOutBack: 'cubic-bezier(0.68, -0.6, 0.32, 1.6)',
      spring: 'cubic-bezier(0.5, 1.5, 0.5, 1)',
      bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    },
  },
  
  // Breakpoints
  breakpoints: {
    xs: '475px',
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
  
  // Z-index layers
  zIndex: {
    base: 0,
    dropdown: 1000,
    sticky: 1100,
    fixed: 1200,
    modalBackdrop: 1300,
    modal: 1400,
    popover: 1500,
    tooltip: 1600,
    notification: 1700,
    max: 9999,
  },
}

// Helper function to get CSS variables from tokens
export const getCSSVariables = (theme: 'light' | 'dark' = 'light') => {
  const variables: Record<string, string> = {}
  
  // Add theme-specific color mappings
  if (theme === 'light') {
    variables['--color-background'] = tokens.colors.neutral[50]
    variables['--color-foreground'] = tokens.colors.neutral[900]
    variables['--color-card'] = tokens.colors.neutral[50]
    variables['--color-card-foreground'] = tokens.colors.neutral[900]
    variables['--color-primary'] = tokens.colors.primary[600]
    variables['--color-primary-foreground'] = tokens.colors.neutral[50]
    variables['--color-secondary'] = tokens.colors.neutral[100]
    variables['--color-secondary-foreground'] = tokens.colors.neutral[900]
    variables['--color-muted'] = tokens.colors.neutral[100]
    variables['--color-muted-foreground'] = tokens.colors.neutral[500]
    variables['--color-accent'] = tokens.colors.primary[100]
    variables['--color-accent-foreground'] = tokens.colors.primary[900]
    variables['--color-destructive'] = tokens.colors.error[500]
    variables['--color-destructive-foreground'] = tokens.colors.neutral[50]
    variables['--color-border'] = tokens.colors.neutral[200]
    variables['--color-input'] = tokens.colors.neutral[200]
    variables['--color-ring'] = tokens.colors.primary[500]
  } else {
    variables['--color-background'] = tokens.colors.neutral[950]
    variables['--color-foreground'] = tokens.colors.neutral[50]
    variables['--color-card'] = tokens.colors.neutral[900]
    variables['--color-card-foreground'] = tokens.colors.neutral[50]
    variables['--color-primary'] = tokens.colors.primary[500]
    variables['--color-primary-foreground'] = tokens.colors.neutral[50]
    variables['--color-secondary'] = tokens.colors.neutral[800]
    variables['--color-secondary-foreground'] = tokens.colors.neutral[50]
    variables['--color-muted'] = tokens.colors.neutral[800]
    variables['--color-muted-foreground'] = tokens.colors.neutral[400]
    variables['--color-accent'] = tokens.colors.neutral[800]
    variables['--color-accent-foreground'] = tokens.colors.neutral[50]
    variables['--color-destructive'] = tokens.colors.error[500]
    variables['--color-destructive-foreground'] = tokens.colors.neutral[50]
    variables['--color-border'] = tokens.colors.neutral[800]
    variables['--color-input'] = tokens.colors.neutral[800]
    variables['--color-ring'] = tokens.colors.primary[400]
  }
  
  return variables
}