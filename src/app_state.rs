use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::models::graph::GraphData;
use crate::config::Settings;
use crate::services::file_service::GitHubService;
use crate::services::perplexity_service::PerplexityServiceImpl;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::speech_service::SpeechService;
use crate::utils::websocket_manager::WebSocketManager;
use crate::utils::gpu_compute::GPUCompute;

pub struct AppState {
    pub graph_data: Arc<RwLock<GraphData>>,
    pub file_cache: Arc<RwLock<HashMap<String, String>>>,
    pub settings: Settings,
    pub github_service: Arc<dyn GitHubService + Send + Sync>,
    pub perplexity_service: PerplexityServiceImpl,
    pub ragflow_service: Arc<RAGFlowService>,
    pub speech_service: Arc<SpeechService>,
    pub websocket_manager: Arc<WebSocketManager>,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        graph_data: Arc<RwLock<GraphData>>,
        file_cache: Arc<RwLock<HashMap<String, String>>>,
        settings: Settings,
        github_service: Arc<dyn GitHubService + Send + Sync>,
        perplexity_service: PerplexityServiceImpl,
        ragflow_service: Arc<RAGFlowService>,
        speech_service: Arc<SpeechService>,
        websocket_manager: Arc<WebSocketManager>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    ) -> Self {
        AppState {
            graph_data,
            file_cache,
            settings,
            github_service,
            perplexity_service,
            ragflow_service,
            speech_service,
            websocket_manager,
            gpu_compute,
        }
    }
}