declare module 'd3-force-3d' {
  export interface SimulationNodeDatum {
    x?: number;
    y?: number;
    z?: number;
    vx?: number;
    vy?: number;
    vz?: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
    index?: number;
  }

  export interface SimulationLinkDatum<NodeDatum extends SimulationNodeDatum> {
    source: string | NodeDatum;
    target: string | NodeDatum;
    index?: number;
  }

  export interface ForceLink<NodeDatum extends SimulationNodeDatum> {
    links(): SimulationLinkDatum<NodeDatum>[];
    links(links: SimulationLinkDatum<NodeDatum>[]): this;
    id(): (node: NodeDatum) => string;
    id(id: (node: NodeDatum) => string): this;
    distance(): number;
    distance(distance: number): this;
    strength(): number;
    strength(strength: number): this;
  }

  export interface ForceManyBody<NodeDatum extends SimulationNodeDatum> {
    strength(): number;
    strength(strength: number): this;
  }

  export interface ForceCenter {
    x(): number;
    x(x: number): this;
    y(): number;
    y(y: number): this;
    z(): number;
    z(z: number): this;
  }

  export interface Simulation<NodeDatum extends SimulationNodeDatum> {
    nodes(): NodeDatum[];
    nodes(nodes: NodeDatum[]): this;
    alpha(): number;
    alpha(alpha: number): this;
    alphaMin(): number;
    alphaMin(min: number): this;
    alphaDecay(): number;
    alphaDecay(decay: number): this;
    alphaTarget(): number;
    alphaTarget(target: number): this;
    velocityDecay(): number;
    velocityDecay(decay: number): this;
    force<T>(name: string): T | null;
    force<T>(name: string, force: null): this;
    force<T>(name: string, force: T): this;
    find(x: number, y: number, z: number, radius?: number): NodeDatum | undefined;
    on(typenames: string, listener: (this: any, ...args: any[]) => void): this;
    tick(): void;
    stop(): this;
    restart(): this;
  }

  export function forceSimulation<NodeDatum extends SimulationNodeDatum>(nodes?: NodeDatum[]): Simulation<NodeDatum>;
  
  export function forceLink<NodeDatum extends SimulationNodeDatum>(): ForceLink<NodeDatum>;
  
  export function forceManyBody<NodeDatum extends SimulationNodeDatum>(): ForceManyBody<NodeDatum>;
  
  export function forceCenter(): ForceCenter;
}
