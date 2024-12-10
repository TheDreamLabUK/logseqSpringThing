use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

use crate::config::Settings;
use crate::services::file_service::RealGitHubService;
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::speech_service::SpeechService;
use crate::services::graph_service::GraphService;
use crate::services::github_service::RealGitHubPRService;
use crate::utils::gpu_compute::GPUCompute;
use crate::models::metadata::Metadata;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<RwLock<Settings>>,
    pub github_service: Arc<RealGitHubService>,
    pub perplexity_service: Arc<PerplexityService>,
    pub ragflow_service: Arc<RAGFlowService>,
    pub speech_service: Arc<RwLock<Option<Arc<SpeechService>>>>,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    pub graph_service: GraphService,
    pub metadata: Arc<RwLock<HashMap<String, Metadata>>>,
    pub ragflow_conversation_id: String,
    pub github_pr_service: Arc<RealGitHubPRService>,
}

impl AppState {
    pub fn new(
        settings: Arc<RwLock<Settings>>,
        github_service: Arc<RealGitHubService>,
        perplexity_service: Arc<PerplexityService>,
        ragflow_service: Arc<RAGFlowService>,
        speech_service: Option<Arc<SpeechService>>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        ragflow_conversation_id: String,
        github_pr_service: Arc<RealGitHubPRService>,
    ) -> Self {
        Self {
            settings,
            github_service,
            perplexity_service,
            ragflow_service,
            speech_service: Arc::new(RwLock::new(speech_service)),
            gpu_compute,
            graph_service: GraphService::new(),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            ragflow_conversation_id,
            github_pr_service,
        }
    }

    pub async fn set_speech_service(&self, service: Arc<SpeechService>) {
        let mut speech_service = self.speech_service.write().await;
        *speech_service = Some(service);
    }

    pub async fn get_speech_service(&self) -> Option<Arc<SpeechService>> {
        self.speech_service.read().await.clone()
    }
}
