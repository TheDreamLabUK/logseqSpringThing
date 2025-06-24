import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "../utils/utils" // Corrected path

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> & {
    label?: string
  }
>(({ className, label, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center",
      className
    )}
    aria-label={label}
    {...props}
  >
    {/* Track: use bg-muted for better contrast */}
    <SliderPrimitive.Track className="custom-slider-track relative h-2 w-full grow overflow-hidden rounded-full bg-muted">
      <SliderPrimitive.Range className="custom-slider-range absolute h-full bg-primary" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb 
      className="custom-slider-thumb block h-5 w-5 rounded-full border-2 border-primary bg-primary shadow-lg ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
      aria-label={label ? `${label} slider thumb` : "Slider thumb"}
    />
  </SliderPrimitive.Root>
))
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }