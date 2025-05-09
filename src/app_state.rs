use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use tokio::sync::RwLock;
use actix_web::web;
use log::info;

use crate::handlers::socket_flow_handler::ClientManager;
use crate::config::AppFullSettings; // Renamed for clarity, ClientFacingSettings removed
use tokio::time::Duration;
use crate::config::feature_access::FeatureAccess;
use crate::models::metadata::MetadataStore;
use crate::models::protected_settings::{ProtectedSettings, ApiKeys, NostrUser};
use crate::services::graph_service::GraphService;
use crate::services::github::{GitHubClient, ContentAPI};
use crate::services::perplexity_service::PerplexityService;
use crate::services::speech_service::SpeechService;
use crate::services::ragflow_service::RAGFlowService;
use crate::services::nostr_service::NostrService;
use crate::utils::gpu_compute::GPUCompute;
use once_cell::sync::Lazy;

#[derive(Clone)]
pub struct AppState {
    pub graph_service: GraphService,
    pub gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    pub settings: Arc<RwLock<AppFullSettings>>, // Changed to AppFullSettings
    pub protected_settings: Arc<RwLock<ProtectedSettings>>,
    pub metadata: Arc<RwLock<MetadataStore>>,
    pub github_client: Arc<GitHubClient>,
    pub content_api: Arc<ContentAPI>,
    pub perplexity_service: Option<Arc<PerplexityService>>,
    pub ragflow_service: Option<Arc<RAGFlowService>>,
    pub speech_service: Option<Arc<SpeechService>>,
    pub nostr_service: Option<web::Data<NostrService>>,
    pub feature_access: web::Data<FeatureAccess>,
    pub ragflow_session_id: String,
    pub active_connections: Arc<AtomicUsize>,
    // pub client_manager: Option<Arc<ClientManager>>, // Removed, use ensure_client_manager
}

static APP_CLIENT_MANAGER: Lazy<Arc<ClientManager>> =
    Lazy::new(|| Arc::new(ClientManager::new()));

impl AppState {
    pub async fn new(
        settings: Arc<RwLock<AppFullSettings>>, // Changed to AppFullSettings
        github_client: Arc<GitHubClient>,
        content_api: Arc<ContentAPI>,
        perplexity_service: Option<Arc<PerplexityService>>,
        ragflow_service: Option<Arc<RAGFlowService>>,
        speech_service: Option<Arc<SpeechService>>,
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        ragflow_session_id: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!("[AppState::new] Initializing GraphService");
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        // GraphService::new might need adjustment if it expects client-facing Settings
        // For now, passing AppFullSettings. This will likely require GraphService changes.
        let graph_service = GraphService::new(settings.clone(), gpu_compute.clone(), APP_CLIENT_MANAGER.clone()).await;
        info!("[AppState::new] GraphService initialization complete");
        
        Ok(Self {
            graph_service,
            gpu_compute,
            settings, // Storing AppFullSettings
            protected_settings: Arc::new(RwLock::new(ProtectedSettings::default())),
            metadata: Arc::new(RwLock::new(MetadataStore::new())),
            github_client,
            content_api,
            perplexity_service,
            ragflow_service,
            speech_service,
            nostr_service: None,
            feature_access: web::Data::new(FeatureAccess::from_env()),
            ragflow_session_id,
            active_connections: Arc::new(AtomicUsize::new(0)),
            // client_manager: Some(APP_CLIENT_MANAGER.clone()), // Removed
        })
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
    
    pub async fn ensure_client_manager(&self) -> Arc<ClientManager> {
        APP_CLIENT_MANAGER.clone()
    }
}
