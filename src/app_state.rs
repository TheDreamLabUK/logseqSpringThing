use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Settings;
use crate::models::metadata::MetadataStore;
use crate::services::graph_service::GraphService;
use crate::services::file_service::RealGitHubService;
use crate::services::github_service::RealGitHubPRService;
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::RAGFlowService;
use crate::utils::gpu_compute::GPUCompute;

#[derive(Clone)]
pub struct AppState {
    pub graph_service: GraphService,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    pub settings: Arc<RwLock<Settings>>,
    pub metadata: Arc<RwLock<MetadataStore>>,
    pub github_service: Arc<RealGitHubService>,
    pub perplexity_service: Arc<PerplexityService>,
    pub ragflow_service: Arc<RAGFlowService>,
    pub ragflow_conversation_id: String,
    pub github_pr_service: Arc<RealGitHubPRService>,
}

impl AppState {
    pub fn new(
        settings: Arc<RwLock<Settings>>,
        github_service: Arc<RealGitHubService>,
        perplexity_service: Arc<PerplexityService>,
        ragflow_service: Arc<RAGFlowService>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        ragflow_conversation_id: String,
        github_pr_service: Arc<RealGitHubPRService>,
    ) -> Self {
        Self {
            graph_service: GraphService::new(),
            gpu_compute,
            settings,
            metadata: Arc::new(RwLock::new(MetadataStore::new())),
            github_service,
            perplexity_service,
            ragflow_service,
            ragflow_conversation_id,
            github_pr_service,
        }
    }
}
