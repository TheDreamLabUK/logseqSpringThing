//! GitHub service facade providing access to all GitHub operations
//! This module re-exports the functionality from the github module
//! in a way that maintains compatibility with existing code

use crate::config::Settings;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::error::Error;
use chrono::{DateTime, Utc};

pub use super::github::types::{GitHubError, GitHubFile, GitHubFileMetadata};
use super::github::{GitHubClient, ContentAPI, PullRequestAPI};

/// Service for interacting with GitHub APIs
pub struct GitHubService {
    client: GitHubClient,
}

impl GitHubService {
    pub fn new(
        token: String,
        owner: String,
        repo: String,
        base_path: String,
        settings: Arc<RwLock<Settings>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = GitHubClient::new(token, owner, repo, base_path, settings)?;
        Ok(Self { client })
    }

    /// Get file content operations
    pub fn content(&self) -> ContentAPI {
        ContentAPI::new(&self.client)
    }

    /// Get pull request operations
    pub fn pull_requests(&self) -> PullRequestAPI {
        PullRequestAPI::new(&self.client)
    }
}

// Implement the old trait for backward compatibility
#[async_trait::async_trait]
pub trait GitHubPRService: Send + Sync {
    async fn create_pull_request(
        &self,
        file_name: &str,
        content: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>>;
}

#[async_trait::async_trait]
impl GitHubPRService for GitHubService {
    async fn create_pull_request(
        &self,
        file_name: &str,
        content: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        self.pull_requests().create_pull_request(file_name, content, original_sha).await
    }
}
