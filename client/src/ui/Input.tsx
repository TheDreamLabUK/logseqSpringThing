import * as React from "react"

import { cn } from "../utils/utils" // Corrected path

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean
  errorMessage?: string
  label?: string
  helperText?: string
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, error, errorMessage, label, helperText, id, required, ...props }, ref) => {
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`
    const errorId = `${inputId}-error`
    const helperId = `${inputId}-helper`
    
    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium mb-1">
            {label}
            {required && <span className="text-destructive ml-1" aria-label="required">*</span>}
          </label>
        )}
        <input
          id={inputId}
          type={type}
          className={cn(
            "flex h-9 w-full rounded-md border bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
            error ? "border-destructive focus-visible:ring-destructive" : "border-input",
            className
          )}
          ref={ref}
          aria-invalid={error}
          aria-describedby={cn(
            error && errorMessage ? errorId : undefined,
            helperText ? helperId : undefined
          )}
          aria-required={required}
          required={required}
          {...props}
        />
        {errorMessage && error && (
          <p id={errorId} className="text-sm text-destructive mt-1" role="alert">
            {errorMessage}
          </p>
        )}
        {helperText && !error && (
          <p id={helperId} className="text-sm text-muted-foreground mt-1">
            {helperText}
          </p>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"

export { Input }