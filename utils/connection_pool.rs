use deadpool_postgres::{Pool, Manager, ManagerConfig, RecyclingMethod};
use tokio_postgres::{NoTls, Config};
use std::str::FromStr;
use std::time::Duration;

pub struct ConnectionPoolManager {
    pool: Pool,
    max_connections: u32,
    connection_timeout: Duration,
}

impl ConnectionPoolManager {
    pub async fn new(database_url: &str, max_connections: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = tokio_postgres::Config::from_str(database_url)?;
        config.connect_timeout(Duration::from_secs(5));
        
        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let mgr = Manager::from_config(config, NoTls, mgr_config);
        let pool = Pool::builder(mgr)
            .max_size(max_connections as usize)
            .build()?;

        Ok(Self {
            pool,
            max_connections,
            connection_timeout: Duration::from_secs(30),
        })
    }

    pub async fn get_connection(&self) -> Result<deadpool_postgres::Client, deadpool_postgres::PoolError> {
        self.pool.get().await
    }

    pub fn get_pool_status(&self) -> PoolStatus {
        PoolStatus {
            total_connections: self.pool.status().size,
            available_connections: self.pool.status().available,
            max_connections: self.max_connections,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoolStatus {
    pub total_connections: u32,
    pub available_connections: u32,
    pub max_connections: u32,
} 