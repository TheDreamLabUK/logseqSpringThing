import React, { useState, useEffect, useRef } from 'react';
import { Search, X } from 'lucide-react';
import { cn } from '../utils/cn';

interface SearchInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'value' | 'type'> {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  autoFocus?: boolean;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  label?: string;
}

export function SearchInput({
  value,
  onChange,
  placeholder = "Search...",
  className,
  autoFocus = false,
  onKeyDown,
  label,
  ...props
}: SearchInputProps) {
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  const handleClear = () => {
    onChange('');
    inputRef.current?.focus();
  };

  return (
    <div className={cn(
      "relative flex items-center",
      className
    )}>
      <Search 
        aria-hidden="true"
        className={cn(
          "absolute left-3 h-4 w-4 transition-colors",
          isFocused ? "text-primary" : "text-muted-foreground"
        )} 
      />
      
      <input
        ref={inputRef}
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        aria-label={label || placeholder}
        className={cn(
          "w-full h-9 pl-9 pr-9 text-sm rounded-md",
          "bg-background border transition-all",
          "placeholder:text-muted-foreground",
          "focus:outline-none focus:ring-2 focus:ring-primary/20",
          isFocused ? "border-primary" : "border-input",
          "disabled:cursor-not-allowed disabled:opacity-50"
        )}
        {...props}
      />
      
      {value && (
        <button
          type="button"
          onClick={handleClear}
          className={cn(
            "absolute right-2 p-1 rounded-sm",
            "text-muted-foreground hover:text-foreground",
            "hover:bg-muted/50 transition-colors",
            "focus:outline-none focus:ring-2 focus:ring-primary/20"
          )}
          aria-label="Clear search"
        >
          <X className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}