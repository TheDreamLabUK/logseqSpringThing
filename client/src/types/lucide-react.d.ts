// client/src/types/lucide-react.d.ts
declare module 'lucide-react' {
  import { FC, SVGProps } from 'react';

  export interface IconProps extends SVGProps<SVGSVGElement> {
    size?: string | number;
    color?: string;
    strokeWidth?: string | number;
  }

  export type Icon = FC<IconProps>;

  const icons: { [key: string]: Icon };
  export default icons;

  // You can still export specific icons if you prefer, but the default export covers all of them.
  export const X: Icon;
  export const Settings: Icon;
  export const Eye: Icon;
  export const ChevronDown: Icon;
  export const ChevronUp: Icon;
  export const Check: Icon;
  export const Send: Icon;
  export const Download: Icon;
  export const Anchor: Icon;
  export const Search: Icon;
  export const Keyboard: Icon;
  export const User: Icon;
  export const Undo: Icon;
  export const Redo: Icon;
  export const History: Icon;
  export const HelpCircle: Icon;
  export const ExternalLink: Icon;
  export const Info: Icon;
  export const Minimize: Icon;
  export const Maximize: Icon;
  export const Terminal: Icon;
  export const Smartphone: Icon;
}