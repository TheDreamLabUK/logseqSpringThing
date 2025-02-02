use reqwest::Client;
use std::time::Duration;
use log::debug;
use super::config::GitHubConfig;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::error::Error;
use crate::config::Settings;

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
    pub async fn new(
        config: GitHubConfig,
        settings: Arc<RwLock<Settings>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let settings_guard = settings.read().await;
        let debug_enabled = settings_guard.system.debug.enabled;
        drop(settings_guard);

        if debug_enabled {
            debug!("Initializing GitHub client - Owner: '{}', Repo: '{}', Base path: '{}'",
                config.owner, config.repo, config.base_path);
        }

        // Build HTTP client with configuration
        if debug_enabled {
            debug!("Configuring HTTP client - Timeout: 30s, User-Agent: github-api-client");
        }

        let client = Client::builder()
            .user_agent("github-api-client")
            .timeout(Duration::from_secs(30))
            .build()?;

        if debug_enabled {
            debug!("HTTP client configured successfully");
        }

        // First decode any existing encoding
        let decoded_path = urlencoding::decode(&config.base_path)
            .unwrap_or(std::borrow::Cow::Owned(config.base_path.clone()))
            .into_owned();
        
        if debug_enabled {
            debug!("Decoded base path: '{}'", decoded_path);
        }
        
        // Clean the path
        let base_path = decoded_path
            .trim_matches('/')
            .replace("//", "/")
            .replace('\\', "/");
        
        if debug_enabled {
            debug!("Cleaned base path: '{}' (original: '{}')", base_path, base_path);
            debug!("GitHub client initialization complete");
        }

        Ok(Self {
            client,
            token: config.token,
            owner: config.owner,
            repo: config.repo,
            base_path,
            settings: Arc::clone(&settings),
        })
    }

    /// Get the properly encoded API path
    pub(crate) async fn get_api_path(&self) -> String {
        let settings = self.settings.read().await;
        let debug_enabled = settings.system.debug.enabled;
        drop(settings);

        if debug_enabled {
            debug!("Getting API path from base_path: '{}'", self.base_path);
        }

        let decoded_path = urlencoding::decode(&self.base_path)
            .unwrap_or(std::borrow::Cow::Owned(self.base_path.clone()))
            .into_owned();

        if debug_enabled {
            log::debug!("Decoded base path: '{}'", decoded_path);
        }

        let trimmed_path = decoded_path.trim_matches('/');
        
        if debug_enabled {
            log::debug!("Trimmed path: '{}'", trimmed_path);
        }
        
        if trimmed_path.is_empty() {
            if debug_enabled {
                log::debug!("Path is empty, returning empty string");
            }
            String::new()
        } else {
            let encoded = url::form_urlencoded::byte_serialize(trimmed_path.as_bytes())
                .collect::<String>();
            
            if debug_enabled {
                log::debug!("Final encoded API path: '{}'", encoded);
            }
            encoded
        }
    }

    /// Get the full path for a file
    pub(crate) async fn get_full_path(&self, path: &str) -> String {
        let settings = self.settings.read().await;
        let debug_enabled = settings.system.debug.enabled;
        drop(settings);

        if debug_enabled {
            debug!("Getting full path - Base: '{}', Input path: '{}'",
                self.base_path, path);
        }

        let base = self.base_path.trim_matches('/');
        let path = path.trim_matches('/');

        if debug_enabled {
            log::debug!("Trimmed paths - Base: '{}', Path: '{}'", base, path);
        }
        
        // First decode any existing encoding to prevent double-encoding
        let decoded_path = urlencoding::decode(path)
            .unwrap_or(std::borrow::Cow::Owned(path.to_string()))
            .into_owned();
        let decoded_base = urlencoding::decode(base)
            .unwrap_or(std::borrow::Cow::Owned(base.to_string()))
            .into_owned();
        
        if debug_enabled {
            log::debug!("Decoded paths - Base: '{}', Path: '{}'",
                decoded_base, decoded_path);
        }
        
        let full_path = if !decoded_base.is_empty() {
            if decoded_path.is_empty() {
                if debug_enabled {
                    log::debug!("Using base path only: '{}'", decoded_base);
                }
                decoded_base
            } else {
                let combined = format!("{}/{}", decoded_base, decoded_path);
                if debug_enabled {
                    log::debug!("Combined path: '{}'", combined);
                }
                combined
            }
        } else {
            if debug_enabled {
                log::debug!("Using decoded path only: '{}'", decoded_path);
            }
            decoded_path
        };

        let encoded = url::form_urlencoded::byte_serialize(full_path.as_bytes())
            .collect::<String>();

        if debug_enabled {
            log::debug!("Final encoded full path: '{}'", encoded);
        }

        encoded
    }

    /// Get the base URL for contents API
    pub(crate) async fn get_contents_url(&self, path: &str) -> String {
        let settings = self.settings.read().await;
        let debug_enabled = settings.system.debug.enabled;
        drop(settings);

        if debug_enabled {
            debug!("Constructing contents URL - Owner: '{}', Repo: '{}', Path: '{}'",
                self.owner, self.repo, path);
        }

        let full_path = self.get_full_path(path).await;
        
        if debug_enabled {
            debug!("Encoded full path: '{}'", full_path);
        }

        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner,
            self.repo,
            full_path
        );

        if debug_enabled {
            debug!("Final contents URL: '{}'", url);
        }

        url
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