use actix_web::{dev::ServiceRequest, Error};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub struct RateLimiter {
    requests: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    window: Duration,
    max_requests: usize,
}

impl RateLimiter {
    pub fn new(window_secs: u64, max_requests: usize) -> Self {
        Self {
            requests: Arc::new(RwLock::new(HashMap::new())),
            window: Duration::from_secs(window_secs),
            max_requests,
        }
    }

    pub async fn check_rate_limit(&self, ip: &str) -> bool {
        let now = Instant::now();
        let mut requests = self.requests.write().await;
        
        let timestamps = requests.entry(ip.to_string()).or_insert_with(Vec::new);
        timestamps.retain(|&t| now.duration_since(t) < self.window);
        
        if timestamps.len() >= self.max_requests {
            false
        } else {
            timestamps.push(now);
            true
        }
    }
} 