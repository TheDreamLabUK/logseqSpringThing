/**
 * Enhanced Theme Provider
 * Manages theme state and provides design system context
 */

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getCSSVariables, tokens } from './tokens'
import { animations } from './animations'

type Theme = 'light' | 'dark' | 'system'
type ResolvedTheme = 'light' | 'dark'

interface ThemeContextValue {
  theme: Theme
  resolvedTheme: ResolvedTheme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  systemTheme: ResolvedTheme
}

const ThemeContext = React.createContext<ThemeContextValue | undefined>(undefined)

export const useTheme = () => {
  const context = React.useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

interface ThemeProviderProps {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
  enableSystem?: boolean
  disableTransitionOnChange?: boolean
}

export const ThemeProvider = ({
  children,
  defaultTheme = 'system',
  storageKey = 'theme',
  enableSystem = true,
  disableTransitionOnChange = false,
}: ThemeProviderProps) => {
  const [theme, setThemeState] = React.useState<Theme>(defaultTheme)
  const [systemTheme, setSystemTheme] = React.useState<ResolvedTheme>('light')
  const [mounted, setMounted] = React.useState(false)
  
  // Get resolved theme (actual light/dark)
  const resolvedTheme = React.useMemo<ResolvedTheme>(() => {
    if (theme === 'system' && enableSystem) {
      return systemTheme
    }
    return theme as ResolvedTheme
  }, [theme, systemTheme, enableSystem])
  
  // Load theme from localStorage
  React.useEffect(() => {
    const savedTheme = localStorage.getItem(storageKey) as Theme | null
    if (savedTheme) {
      setThemeState(savedTheme)
    }
    setMounted(true)
  }, [storageKey])
  
  // Detect system theme
  React.useEffect(() => {
    if (!enableSystem) return
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    setSystemTheme(mediaQuery.matches ? 'dark' : 'light')
    
    const handleChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light')
    }
    
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [enableSystem])
  
  // Apply theme to document
  React.useEffect(() => {
    if (!mounted) return
    
    const root = document.documentElement
    const cssVariables = getCSSVariables(resolvedTheme)
    
    // Apply theme class
    root.classList.remove('light', 'dark')
    root.classList.add(resolvedTheme)
    
    // Apply CSS variables
    Object.entries(cssVariables).forEach(([key, value]) => {
      root.style.setProperty(key, value)
    })
    
    // Handle transition
    if (disableTransitionOnChange) {
      root.classList.add('disable-transitions')
      setTimeout(() => {
        root.classList.remove('disable-transitions')
      }, 0)
    }
    
    // Update meta theme-color
    const metaThemeColor = document.querySelector('meta[name="theme-color"]')
    if (metaThemeColor) {
      metaThemeColor.setAttribute(
        'content',
        resolvedTheme === 'dark' ? tokens.colors.neutral[950] : tokens.colors.neutral[50]
      )
    }
  }, [resolvedTheme, mounted, disableTransitionOnChange])
  
  const setTheme = React.useCallback(
    (newTheme: Theme) => {
      setThemeState(newTheme)
      localStorage.setItem(storageKey, newTheme)
    },
    [storageKey]
  )
  
  const toggleTheme = React.useCallback(() => {
    setTheme(resolvedTheme === 'light' ? 'dark' : 'light')
  }, [resolvedTheme, setTheme])
  
  const value = React.useMemo(
    () => ({
      theme,
      resolvedTheme,
      setTheme,
      toggleTheme,
      systemTheme,
    }),
    [theme, resolvedTheme, setTheme, toggleTheme, systemTheme]
  )
  
  // Prevent flash of incorrect theme
  if (!mounted) {
    return null
  }
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}

// Theme Toggle Component
interface ThemeToggleProps {
  className?: string
  showLabel?: boolean
  size?: 'sm' | 'md' | 'lg'
}

export const ThemeToggle = ({ className, showLabel = false, size = 'md' }: ThemeToggleProps) => {
  const { theme, setTheme, resolvedTheme } = useTheme()
  
  const sizes = {
    sm: 'h-8 w-8',
    md: 'h-10 w-10',
    lg: 'h-12 w-12',
  }
  
  const iconSizes = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
  }
  
  const options: { value: Theme; icon: React.ReactNode; label: string }[] = [
    {
      value: 'light',
      label: 'Light',
      icon: (
        <svg className={iconSizes[size]} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
          />
        </svg>
      ),
    },
    {
      value: 'dark',
      label: 'Dark',
      icon: (
        <svg className={iconSizes[size]} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
          />
        </svg>
      ),
    },
    {
      value: 'system',
      label: 'System',
      icon: (
        <svg className={iconSizes[size]} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
          />
        </svg>
      ),
    },
  ]
  
  const currentOption = options.find((opt) => opt.value === theme) || options[0]
  
  return (
    <div className={className}>
      {showLabel && (
        <label className="text-sm font-medium text-muted-foreground mb-2 block">
          Theme
        </label>
      )}
      <div className="flex gap-1 p-1 bg-muted rounded-lg">
        {options.map((option) => (
          <motion.button
            key={option.value}
            onClick={() => setTheme(option.value)}
            className={`
              relative ${sizes[size]} rounded-md flex items-center justify-center
              transition-colors duration-200
              ${theme === option.value
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
              }
            `}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={animations.transitions.spring.snappy}
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={option.value}
                initial={{ opacity: 0, rotate: -90, scale: 0.8 }}
                animate={{ opacity: 1, rotate: 0, scale: 1 }}
                exit={{ opacity: 0, rotate: 90, scale: 0.8 }}
                transition={animations.transitions.spring.bounce}
              >
                {option.icon}
              </motion.div>
            </AnimatePresence>
            <span className="sr-only">{option.label}</span>
          </motion.button>
        ))}
      </div>
    </div>
  )
}

// Theme-aware component wrapper
interface ThemedProps {
  children: React.ReactNode
  className?: string
  lightClassName?: string
  darkClassName?: string
}

export const Themed = ({ children, className, lightClassName, darkClassName }: ThemedProps) => {
  const { resolvedTheme } = useTheme()
  
  const finalClassName = React.useMemo(() => {
    const classes = [className]
    if (resolvedTheme === 'light' && lightClassName) {
      classes.push(lightClassName)
    } else if (resolvedTheme === 'dark' && darkClassName) {
      classes.push(darkClassName)
    }
    return classes.filter(Boolean).join(' ')
  }, [className, lightClassName, darkClassName, resolvedTheme])
  
  return <div className={finalClassName}>{children}</div>
}