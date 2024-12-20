use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
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
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub ragflow_conversation_id: String,
    pub github_pr_service: Arc<RealGitHubPRService>,
    pub active_connections: Arc<AtomicUsize>,
}

impl AppState {
    pub fn new(
        settings: Arc<RwLock<Settings>>,
        github_service: Arc<RealGitHubService>,
        perplexity_service: Option<Arc<PerplexityService>>,
        ragflow_service: Option<Arc<RAGFlowService>>,
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
            active_connections: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn increment_connections(&self) -> usize {
        self.active_connections.fetch_add(1, Ordering::SeqCst)
    }

    pub fn decrement_connections(&self) -> usize {
        self.active_connections.fetch_sub(1, Ordering::SeqCst)
    }
}
