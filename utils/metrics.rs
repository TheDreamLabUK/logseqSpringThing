use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub fps: f64,
    pub frame_time_ms: f64,
    pub gpu_compute_time_ms: f64,
    pub node_count: usize,
    pub edge_count: usize,
    pub memory_usage_mb: f64,
    pub websocket_connections: usize,
}

#[derive(Debug)]
pub struct MetricsCollector {
    frame_times: RwLock<VecDeque<Duration>>,
    gpu_compute_times: RwLock<VecDeque<Duration>>,
    node_count: AtomicUsize,
    edge_count: AtomicUsize,
    websocket_connections: AtomicUsize,
    last_memory_check: RwLock<(Instant, f64)>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            frame_times: RwLock::new(VecDeque::with_capacity(120)), // 2 seconds at 60fps
            gpu_compute_times: RwLock::new(VecDeque::with_capacity(120)),
            node_count: AtomicUsize::new(0),
            edge_count: AtomicUsize::new(0),
            websocket_connections: AtomicUsize::new(0),
            last_memory_check: RwLock::new((Instant::now(), 0.0)),
        }
    }

    pub async fn record_frame_time(&self, duration: Duration) {
        let mut times = self.frame_times.write().await;
        if times.len() >= 120 {
            times.pop_front();
        }
        times.push_back(duration);
    }

    pub async fn record_gpu_compute_time(&self, duration: Duration) {
        let mut times = self.gpu_compute_times.write().await;
        if times.len() >= 120 {
            times.pop_front();
        }
        times.push_back(duration);
    }

    pub fn update_graph_stats(&self, nodes: usize, edges: usize) {
        self.node_count.store(nodes, Ordering::Relaxed);
        self.edge_count.store(edges, Ordering::Relaxed);
    }

    pub fn update_connection_count(&self, count: usize) {
        self.websocket_connections.store(count, Ordering::Relaxed);
    }

    async fn get_memory_usage(&self) -> f64 {
        let mut last_check = self.last_memory_check.write().await;
        let now = Instant::now();

        // Only update memory usage every 5 seconds
        if now.duration_since(last_check.0) > Duration::from_secs(5) {
            #[cfg(target_os = "linux")]
            {
                if let Ok(mem_info) = std::fs::read_to_string("/proc/self/statm") {
                    if let Some(resident_pages) = mem_info.split_whitespace().nth(1) {
                        if let Ok(pages) = resident_pages.parse::<u64>() {
                            let page_size = 4096; // Standard page size
                            last_check.1 = (pages * page_size) as f64 / (1024.0 * 1024.0);
                        }
                    }
                }
            }
            last_check.0 = now;
        }

        last_check.1
    }

    pub async fn get_metrics(&self) -> PerformanceMetrics {
        let frame_times = self.frame_times.read().await;
        let gpu_times = self.gpu_compute_times.read().await;

        let avg_frame_time = if !frame_times.is_empty() {
            frame_times.iter().sum::<Duration>().as_secs_f64() / frame_times.len() as f64
        } else {
            0.0
        };

        let avg_gpu_time = if !gpu_times.is_empty() {
            gpu_times.iter().sum::<Duration>().as_secs_f64() / gpu_times.len() as f64
        } else {
            0.0
        };

        PerformanceMetrics {
            fps: if avg_frame_time > 0.0 { 1.0 / avg_frame_time } else { 0.0 },
            frame_time_ms: avg_frame_time * 1000.0,
            gpu_compute_time_ms: avg_gpu_time * 1000.0,
            node_count: self.node_count.load(Ordering::Relaxed),
            edge_count: self.edge_count.load(Ordering::Relaxed),
            memory_usage_mb: self.get_memory_usage().await,
            websocket_connections: self.websocket_connections.load(Ordering::Relaxed),
        }
    }
} 