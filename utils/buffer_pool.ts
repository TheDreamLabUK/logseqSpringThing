import { Logger } from './logger';

const log = new Logger('BufferPool');

interface PoolConfig {
    maxPoolSize: number;
    cleanupInterval: number;
    bufferTTL: number;
}

interface PoolStats {
    hits: number;
    misses: number;
    totalAllocated: number;
    lastCleanup: number;
}

export class BufferPool {
    private pools: Map<number, ArrayBuffer[]> = new Map();
    private stats: PoolStats = {
        hits: 0,
        misses: 0,
        totalAllocated: 0,
        lastCleanup: Date.now(),
    };
    private config: PoolConfig;

    constructor(config?: Partial<PoolConfig>) {
        this.config = {
            maxPoolSize: 100,
            cleanupInterval: 60000, // 1 minute
            bufferTTL: 300000, // 5 minutes
            ...config,
        };
    }

    public async getBuffer(size: number): Promise<ArrayBuffer> {
        const pool = this.pools.get(size);
        if (pool && pool.length > 0) {
            this.stats.hits++;
            return pool.pop()!;
        }

        this.stats.misses++;
        this.stats.totalAllocated += size;
        return new ArrayBuffer(size);
    }

    public async returnBuffer(buffer: ArrayBuffer): Promise<void> {
        const size = buffer.byteLength;
        let pool = this.pools.get(size);
        
        if (!pool) {
            pool = [];
            this.pools.set(size, pool);
        }

        if (pool.length < this.config.maxPoolSize) {
            pool.push(buffer);
        }

        // Perform cleanup if needed
        if (Date.now() - this.stats.lastCleanup > this.config.cleanupInterval) {
            await this.cleanup();
        }
    }

    private async cleanup(): Promise<void> {
        for (const [size, pool] of this.pools.entries()) {
            if (pool.length > this.config.maxPoolSize) {
                pool.length = this.config.maxPoolSize;
            }
            if (pool.length === 0) {
                this.pools.delete(size);
            }
        }
        this.stats.lastCleanup = Date.now();
    }

    public getStats(): PoolStats {
        return { ...this.stats };
    }

    public dispose(): void {
        this.pools.clear();
        this.stats = {
            hits: 0,
            misses: 0,
            totalAllocated: 0,
            lastCleanup: Date.now(),
        };
    }
} 