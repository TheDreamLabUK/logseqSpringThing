use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationState {
    pub operation_type: String,
    pub started_at: Instant,
    pub last_checkpoint: Instant,
    pub progress: f32,
    pub data: serde_json::Value,
}

pub struct RecoverySystem {
    operations: RwLock<HashMap<String, OperationState>>,
    checkpoint_interval: Duration,
}

impl RecoverySystem {
    pub fn new(checkpoint_interval_secs: u64) -> Self {
        Self {
            operations: RwLock::new(HashMap::new()),
            checkpoint_interval: Duration::from_secs(checkpoint_interval_secs),
        }
    }

    pub async fn start_operation(&self, id: String, op_type: String, data: serde_json::Value) {
        let now = Instant::now();
        let state = OperationState {
            operation_type: op_type,
            started_at: now,
            last_checkpoint: now,
            progress: 0.0,
            data,
        };
        
        let mut ops = self.operations.write().await;
        ops.insert(id, state);
    }

    pub async fn update_progress(&self, id: &str, progress: f32) -> bool {
        let mut ops = self.operations.write().await;
        if let Some(state) = ops.get_mut(id) {
            let now = Instant::now();
            if now.duration_since(state.last_checkpoint) >= self.checkpoint_interval {
                state.last_checkpoint = now;
                state.progress = progress;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub async fn complete_operation(&self, id: &str) {
        let mut ops = self.operations.write().await;
        ops.remove(id);
    }

    pub async fn get_incomplete_operations(&self) -> Vec<(String, OperationState)> {
        let ops = self.operations.read().await;
        ops.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
} 