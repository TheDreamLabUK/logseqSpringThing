use tokio_postgres::{Config, NoTls, Error};
use deadpool_postgres::{Pool, Manager};
use std::str::FromStr;

pub async fn create_pool(database_url: &str) -> Result<Pool, Error> {
    let config = Config::from_str(database_url)?;
    let manager = Manager::new(config, NoTls);
    Ok(Pool::builder(manager)
        .max_size(16)
        .build()
        .unwrap())
} 