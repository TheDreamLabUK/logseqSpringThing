use reqwest::Client;
use std::time::Duration;
use log::{debug, error, info};
use crate::config::Settings;
use std::sync::Arc;
use tokio::sync::RwLock;
use super::types::{GitHubError, GitHubFileMetadata};
use std::error::Error;

const GITHUB_API_DELAY: Duration = Duration::from_millis(500);
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_secs(2);

/// Core GitHub API client providing common functionality
pub struct GitHubClient {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
    settings: Arc<RwLock<Settings>>,
}

impl GitHubClient {
    /// Create a new GitHub API client
    pub fn new(
        token: String,
        owner: String,
        repo: String,
        base_path: String,
        settings: Arc<RwLock<Settings>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        info!("Creating GitHub client with initial base_path: {}", base_path);

        let client = Client::builder()
            .user_agent("github-api-client")
            .timeout(Duration::from_secs(30))
            .build()?;

        // Clean and validate base_path
        let base_path = if base_path.trim().is_empty() {
            info!("Base path was empty, using default: mainKnowledgeGraph/pages");
            "mainKnowledgeGraph/pages".to_string()
        } else {
            // First decode any existing encoding
            let decoded_path = urlencoding::decode(&base_path)
                .unwrap_or(std::borrow::Cow::Owned(base_path.clone()))
                .into_owned();
            
            // Clean the path
            let cleaned_path = decoded_path
                .trim_matches('/')
                .replace("//", "/")
                .replace('\\', "/");
            
            info!("Using cleaned base path: {} (original: {})", cleaned_path, base_path);
            cleaned_path
        };

        Ok(Self {
            client,
            token,
            owner,
            repo,
            base_path,
            settings: Arc::clone(&settings),
        })
    }

    /// Get the properly encoded API path
    pub(crate) fn get_api_path(&self) -> String {
        let decoded_path = urlencoding::decode(&self.base_path)
            .unwrap_or(std::borrow::Cow::Owned(self.base_path.clone()))
            .into_owned();
        let trimmed_path = decoded_path.trim_matches('/');
        
        if trimmed_path.is_empty() {
            String::new()
        } else {
            url::form_urlencoded::byte_serialize(trimmed_path.as_bytes())
                .collect::<String>()
        }
    }

    /// Get the full path for a file
    pub(crate) fn get_full_path(&self, path: &str) -> String {
        let base = self.base_path.trim_matches('/');
        let path = path.trim_matches('/');
        
        // First decode any existing encoding to prevent double-encoding
        let decoded_path = urlencoding::decode(path)
            .unwrap_or(std::borrow::Cow::Owned(path.to_string()))
            .into_owned();
        let decoded_base = urlencoding::decode(base)
            .unwrap_or(std::borrow::Cow::Owned(base.to_string()))
            .into_owned();
        
        let full_path = if !decoded_base.is_empty() {
            if decoded_path.is_empty() {
                decoded_base
            } else {
                format!("{}/{}", decoded_base, decoded_path)
            }
        } else {
            decoded_path
        };

        url::form_urlencoded::byte_serialize(full_path.as_bytes())
            .collect::<String>()
    }

    /// Get the base URL for contents API
    pub(crate) fn get_contents_url(&self, path: &str) -> String {
        format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner,
            self.repo,
            self.get_full_path(path)
        )
    }

    /// Get the client for making requests
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// Get the authorization token
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// Get owner name
    pub(crate) fn owner(&self) -> &str {
        &self.owner
    }

    /// Get repository name
    pub(crate) fn repo(&self) -> &str {
        &self.repo
    }

    /// Get base path
    pub(crate) fn base_path(&self) -> &str {
        &self.base_path
    }

    /// Get settings
    pub(crate) fn settings(&self) -> &Arc<RwLock<Settings>> {
        &self.settings
    }

    /// Get constants
    pub(crate) fn constants() -> (Duration, u32, Duration) {
        (GITHUB_API_DELAY, MAX_RETRIES, RETRY_DELAY)
    }
}