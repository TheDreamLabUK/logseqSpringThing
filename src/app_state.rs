use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::models::graph::GraphData;
use crate::config::Settings;
use crate::services::file_service::GitHubService;

pub struct AppState {
    pub graph_data: Arc<RwLock<GraphData>>,
    pub file_cache: Arc<RwLock<HashMap<String, String>>>,
    pub settings: Settings,
    pub github_service: Arc<dyn GitHubService + Send + Sync>,
}

impl AppState {
    pub fn new(
        graph_data: Arc<RwLock<GraphData>>,
        file_cache: Arc<RwLock<HashMap<String, String>>>,
        settings: Settings,
        github_service: Arc<dyn GitHubService + Send + Sync>
    ) -> Self {
        Self { graph_data, file_cache, settings, github_service }
    }
}
