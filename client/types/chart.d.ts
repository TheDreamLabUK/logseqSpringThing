declare module 'chart.js' {
  export class Chart {
    constructor(ctx: CanvasRenderingContext2D | HTMLCanvasElement, config: ChartConfiguration)
    destroy(): void
    update(config?: { mode?: 'none' | 'default' | 'resize' }): void
    data: ChartData
    options: ChartOptions
  }

  export interface ChartConfiguration {
    type: 'line' | 'bar' | 'radar' | 'doughnut' | 'pie' | 'polarArea' | 'bubble' | 'scatter'
    data: ChartData
    options?: ChartOptions
  }

  export interface ChartData {
    labels: Array<string | number>
    datasets: ChartDataset[]
  }

  export interface ChartDataset {
    label: string
    data: Array<number | { x: number; y: number }>
    borderColor?: string
    backgroundColor?: string
    tension?: number
  }

  export interface ChartOptions {
    responsive?: boolean
    animation?: boolean | { duration: number }
    scales?: {
      x?: {
        type?: string
        time?: {
          unit?: string
        }
      }
      y?: {
        beginAtZero?: boolean
      }
    }
  }

  export class LineController {}
  export class LineElement {}
  export class PointElement {}
  export class LinearScale {}
  export class TimeScale {}
  export class Title {}

  export interface ChartDataPoint {
    x: number
    y: number
  }

  export interface UpdateMode {
    mode?: 'none' | 'default' | 'resize'
  }
}

// Extend window with performance API
declare global {
  interface Window {
    performance: {
      memory?: {
        usedJSHeapSize: number
        totalJSHeapSize: number
        jsHeapSizeLimit: number
      }
    }
  }
}

// Export our custom types
export interface PerformanceChart extends Chart {
  data: {
    labels: number[]
    datasets: Array<{
      label: string
      data: Array<{ x: number; y: number }>
      borderColor: string
      tension: number
    }>
  }
  update(config?: { mode: 'none' | 'default' | 'resize' }): void
}

export type { ChartDataset, ChartDataPoint, ChartOptions } from 'chart.js'
