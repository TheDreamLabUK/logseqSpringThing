use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use tokio::sync::RwLock;
use actix_web::web;

use crate::config::Settings;
use crate::config::feature_access::FeatureAccess;
use crate::models::metadata::MetadataStore;
use crate::models::protected_settings::{ProtectedSettings, ApiKeys, NostrUser};
use crate::services::graph_service::GraphService;
use crate::services::github::{GitHubClient, ContentAPI};
use crate::services::perplexity_service::PerplexityService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::nostr_service::NostrService;
use crate::utils::gpu_compute::GPUCompute;

#[derive(Clone)]
pub struct AppState {
    pub graph_service: GraphService,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    pub settings: Arc<RwLock<Settings>>,
    pub protected_settings: Arc<RwLock<ProtectedSettings>>,
    pub metadata: Arc<RwLock<MetadataStore>>,
    pub github_client: Arc<GitHubClient>,
    pub content_api: Arc<ContentAPI>,
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub nostr_service: Option<web::Data<NostrService>>,
    pub feature_access: web::Data<FeatureAccess>,
    pub ragflow_conversation_id: String,
    pub active_connections: Arc<AtomicUsize>,
}

impl AppState {
    pub fn new(
        settings: Arc<RwLock<Settings>>,
        github_client: Arc<GitHubClient>,
        content_api: Arc<ContentAPI>,
        perplexity_service: Option<Arc<PerplexityService>>,
        ragflow_service: Option<Arc<RAGFlowService>>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        ragflow_conversation_id: String,
    ) -> Self {
        // Initialize FeatureAccess from environment variables
        Self {
            graph_service: GraphService::new(),
            gpu_compute,
            settings,
            protected_settings: Arc::new(RwLock::new(ProtectedSettings::default())),
            metadata: Arc::new(RwLock::new(MetadataStore::new())),
            github_client,
            content_api,
            perplexity_service,
            ragflow_service,
            nostr_service: None,
            feature_access: web::Data::new(FeatureAccess::from_env()),
            ragflow_conversation_id,
            active_connections: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn increment_connections(&self) -> usize {
        self.active_connections.fetch_add(1, Ordering::SeqCst)
    }

    pub fn decrement_connections(&self) -> usize {
        self.active_connections.fetch_sub(1, Ordering::SeqCst)
    }

    pub async fn get_api_keys(&self, pubkey: &str) -> ApiKeys {
        let protected_settings = self.protected_settings.read().await;
        protected_settings.get_api_keys(pubkey)
    }

    pub async fn get_nostr_user(&self, pubkey: &str) -> Option<NostrUser> {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service.get_user(pubkey).await
        } else {
            None
        }
    }

    pub async fn validate_nostr_session(&self, pubkey: &str, token: &str) -> bool {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service.validate_session(pubkey, token).await
        } else {
            false
        }
    }

    pub async fn update_nostr_user_api_keys(&self, pubkey: &str, api_keys: ApiKeys) -> Result<NostrUser, String> {
        if let Some(nostr_service) = &self.nostr_service {
            nostr_service.update_user_api_keys(pubkey, api_keys)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err("Nostr service not initialized".to_string())
        }
    }

    pub fn set_nostr_service(&mut self, service: NostrService) {
        self.nostr_service = Some(web::Data::new(service));
    }

    // Feature access helper methods
    pub fn is_power_user(&self, pubkey: &str) -> bool {
        self.feature_access.is_power_user(pubkey)
    }

    pub fn can_sync_settings(&self, pubkey: &str) -> bool {
        self.feature_access.can_sync_settings(pubkey)
    }

    pub fn has_feature_access(&self, pubkey: &str, feature: &str) -> bool {
        self.feature_access.has_feature_access(pubkey, feature)
    }

    pub fn get_available_features(&self, pubkey: &str) -> Vec<String> {
        self.feature_access.get_available_features(pubkey)
    }
}
