use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct Cache<T> {
    data: RwLock<HashMap<String, (T, Instant)>>,
    ttl: Duration,
}

impl<T: Clone> Cache<T> {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    pub async fn get(&self, key: &str) -> Option<T> {
        let data = self.data.read().await;
        if let Some((value, timestamp)) = data.get(key) {
            if timestamp.elapsed() < self.ttl {
                return Some(value.clone());
            }
        }
        None
    }

    pub async fn set(&self, key: String, value: T) {
        let mut data = self.data.write().await;
        data.insert(key, (value, Instant::now()));
    }

    pub async fn remove(&self, key: &str) {
        let mut data = self.data.write().await;
        data.remove(key);
    }

    pub async fn clear_expired(&self) {
        let mut data = self.data.write().await;
        data.retain(|_, (_, timestamp)| timestamp.elapsed() < self.ttl);
    }
} 