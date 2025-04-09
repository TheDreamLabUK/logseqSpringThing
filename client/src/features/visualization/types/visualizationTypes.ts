import { XRSettings } from '../../xr/types/xr';

export interface VisualizationSettings {
  nodes: {
    color: string;
    defaultSize: number;
    opacity: number;
  };
  edges: {
    color: string;
    width: number;
    opacity: number;
  };
  xr: XRSettings;
}