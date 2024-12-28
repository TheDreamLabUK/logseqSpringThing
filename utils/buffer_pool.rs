use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

pub struct BufferPool {
    pools: RwLock<HashMap<usize, Vec<Vec<u8>>>>,
    stats: RwLock<PoolStats>,
    config: PoolConfig,
}

#[derive(Debug)]
struct PoolStats {
    hits: usize,
    misses: usize,
    total_allocated: usize,
    last_cleanup: Instant,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_pool_size: usize,
    pub cleanup_interval: Duration,
    pub buffer_ttl: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 100,
            cleanup_interval: Duration::from_secs(60),
            buffer_ttl: Duration::from_secs(300),
        }
    }
}

impl BufferPool {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            stats: RwLock::new(PoolStats {
                hits: 0,
                misses: 0,
                total_allocated: 0,
                last_cleanup: Instant::now(),
            }),
            config,
        }
    }

    pub async fn get_buffer(&self, size: usize) -> Vec<u8> {
        let mut pools = self.pools.write().await;
        let mut stats = self.stats.write().await;

        if let Some(pool) = pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                stats.hits += 1;
                return buffer;
            }
        }

        stats.misses += 1;
        stats.total_allocated += size;
        Vec::with_capacity(size)
    }

    pub async fn return_buffer(&self, buffer: Vec<u8>) {
        let capacity = buffer.capacity();
        let mut pools = self.pools.write().await;

        let pool = pools.entry(capacity).or_insert_with(Vec::new);
        if pool.len() < self.config.max_pool_size {
            pool.push(buffer);
        }

        // Perform cleanup if needed
        let mut stats = self.stats.write().await;
        if stats.last_cleanup.elapsed() > self.config.cleanup_interval {
            self.cleanup(&mut pools).await;
            stats.last_cleanup = Instant::now();
        }
    }

    async fn cleanup(&self, pools: &mut HashMap<usize, Vec<Vec<u8>>>) {
        // Remove excess buffers from each pool
        for pool in pools.values_mut() {
            if pool.len() > self.config.max_pool_size {
                pool.truncate(self.config.max_pool_size);
            }
        }

        // Remove empty pools
        pools.retain(|_, pool| !pool.is_empty());
    }

    pub async fn get_stats(&self) -> (usize, usize, usize) {
        let stats = self.stats.read().await;
        (stats.hits, stats.misses, stats.total_allocated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_buffer_pool() {
        let config = PoolConfig {
            max_pool_size: 2,
            cleanup_interval: Duration::from_secs(1),
            buffer_ttl: Duration::from_secs(5),
        };
        let pool = BufferPool::new(config);

        // Get a buffer
        let buf1 = pool.get_buffer(1024).await;
        assert_eq!(buf1.capacity(), 1024);

        // Return it and get another one
        pool.return_buffer(buf1).await;
        let buf2 = pool.get_buffer(1024).await;
        assert_eq!(buf2.capacity(), 1024);

        // Check stats
        let (hits, misses, total) = pool.get_stats().await;
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(total, 1024);
    }
} 