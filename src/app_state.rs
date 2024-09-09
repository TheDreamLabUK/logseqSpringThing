use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::models::graph::GraphData;

pub struct AppState {
    pub graph_data: Arc<RwLock<GraphData>>,
    pub file_cache: Arc<RwLock<HashMap<String, String>>>,
}

impl AppState {
    pub fn new(graph_data: Arc<RwLock<GraphData>>, file_cache: Arc<RwLock<HashMap<String, String>>>) -> Self {
        Self { graph_data, file_cache }
    }
}
