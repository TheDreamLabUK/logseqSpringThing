use crate::config::Settings;
use crate::models::metadata::MetadataStore;
use crate::services::file_service::GitHubService;
use crate::services::github_service::GitHubPRService;
use crate::services::graph_service::GraphService;
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::speech_service::SpeechService;
use crate::utils::websocket_manager::WebSocketManager;
use crate::utils::gpu_compute::GPUCompute;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct AppState {
    pub settings: Arc<RwLock<Settings>>,
    pub metadata: Arc<RwLock<MetadataStore>>,
    pub github_service: Arc<dyn GitHubService + Send + Sync>,
    pub perplexity_service: Arc<PerplexityService>,
    pub ragflow_service: Arc<RAGFlowService>,
    pub speech_service: Arc<SpeechService>,
    pub websocket_manager: Arc<WebSocketManager>,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    pub ragflow_conversation_id: String,
    pub github_pr_service: Arc<dyn GitHubPRService + Send + Sync>,
    pub graph_service: Arc<GraphService>,
}

impl AppState {
    pub fn new(
        settings: Arc<RwLock<Settings>>,
        github_service: Arc<dyn GitHubService + Send + Sync>,
        perplexity_service: Arc<PerplexityService>,
        ragflow_service: Arc<RAGFlowService>,
        speech_service: Arc<SpeechService>,
        websocket_manager: Arc<WebSocketManager>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        ragflow_conversation_id: String,
        github_pr_service: Arc<dyn GitHubPRService + Send + Sync>,
    ) -> Self {
        let graph_service = Arc::new(GraphService::new());
        let metadata = Arc::new(RwLock::new(MetadataStore::new()));
        
        Self {
            settings,
            metadata,
            github_service,
            perplexity_service,
            ragflow_service,
            speech_service,
            websocket_manager,
            gpu_compute,
            ragflow_conversation_id,
            github_pr_service,
            graph_service,
        }
    }
}
