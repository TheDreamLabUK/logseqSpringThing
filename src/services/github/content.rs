use super::api::GitHubClient;
use super::types::{GitHubFileMetadata, GitHubError, RateLimitInfo};
use chrono::{DateTime, Utc};
use log::{debug, error, info};
use std::error::Error;
use std::sync::Arc;
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Handles GitHub content API operations
#[derive(Clone)]
pub struct ContentAPI {
    client: Arc<GitHubClient>,
    rate_limits: Arc<RwLock<HashMap<String, RateLimitInfo>>>,
}

impl ContentAPI {
    /// Create a new ContentAPI instance
    pub fn new(client: Arc<GitHubClient>) -> Self {
        Self {
            client,
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Extract and update rate limit information from response headers
    async fn update_rate_limits(&self, headers: &HeaderMap) {
        if let (Some(remaining), Some(limit), Some(reset)) = (
            headers.get("x-ratelimit-remaining"),
            headers.get("x-ratelimit-limit"),
            headers.get("x-ratelimit-reset")
        ) {
            let remaining = remaining.to_str().unwrap_or("0").parse().unwrap_or(0);
            let limit = limit.to_str().unwrap_or("0").parse().unwrap_or(0);
            let reset = reset.to_str().unwrap_or("0").parse().unwrap_or(0);
            
            let reset_time = DateTime::from_timestamp(reset, 0)
                .unwrap_or_else(|| Utc::now());

            let info = RateLimitInfo {
                remaining,
                limit,
                reset_time,
            };

            let mut limits = self.rate_limits.write().await;
            limits.insert("core".to_string(), info);
        }
    }

    /// Check if we're rate limited
    async fn check_rate_limit(&self) -> Result<(), GitHubError> {
        let limits = self.rate_limits.read().await;
        if let Some(info) = limits.get("core") {
            if info.remaining == 0 {
                return Err(GitHubError::RateLimitExceeded(info.clone()));
            }
        }
        Ok(())
    }

    /// Check if a file is public by reading just the first line
    pub async fn check_file_public(&self, download_url: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
        // Check rate limits before making request
        self.check_rate_limit().await?;

        let response = self.client.client()
            .get(download_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .header("Range", "bytes=0-100") // More flexible range to handle different line endings
            .send()
            .await?;

        // Update rate limits from response headers
        self.update_rate_limits(response.headers()).await;

        let status = response.status();
        match status.as_u16() {
            200 | 206 => { // Success or Partial Content
                let content = response.text().await?;
                debug!("First line check ({}): '{}'", download_url, content.trim());
                Ok(content.trim().starts_with("public:: true"))
            },
            404 => {
                error!("File not found: {}", download_url);
                Err(Box::new(GitHubError::NotFound(download_url.to_string())))
            },
            416 => { // Range Not Satisfiable
                debug!("File exists but is empty or too small: {}", download_url);
                Ok(false)
            },
            429 => {
                let limits = self.rate_limits.read().await;
                if let Some(info) = limits.get("core") {
                    Err(Box::new(GitHubError::RateLimitExceeded(info.clone())))
                } else {
                    Err("Rate limit exceeded without limit info".into())
                }
            },
            _ => {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                error!("Failed to check file public status. Status: {}, Error: {}", status, error_text);
                Err(Box::new(GitHubError::ApiError(format!("{} - {}", status, error_text))))
            }
        }
    }

    /// Fetch full content of a file
    pub async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        // Check rate limits before making request
        self.check_rate_limit().await?;

        let response = self.client.client()
            .get(download_url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        // Update rate limits from response headers
        self.update_rate_limits(response.headers()).await;

        let status = response.status();
        match status.as_u16() {
            200 => {
                let content = response.text().await?;
                Ok(content)
            },
            404 => {
                error!("File not found: {}", download_url);
                Err(Box::new(GitHubError::NotFound(download_url.to_string())))
            },
            429 => {
                let limits = self.rate_limits.read().await;
                if let Some(info) = limits.get("core") {
                    Err(Box::new(GitHubError::RateLimitExceeded(info.clone())))
                } else {
                    Err("Rate limit exceeded without limit info".into())
                }
            },
            _ => {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                error!("Failed to fetch file content. Status: {}, Error: {}", status, error_text);
                Err(Box::new(GitHubError::ApiError(format!("{} - {}", status, error_text))))
            }
        }
    }

    /// Get the last modified time for a file
    pub async fn get_file_last_modified(&self, file_path: &str) -> Result<DateTime<Utc>, Box<dyn Error + Send + Sync>> {
        // Check rate limits before making request
        self.check_rate_limit().await?;

        // Use GitHubClient's path handling
        let encoded_path = self.client.get_full_path(file_path);
        let url = format!(
            "https://api.github.com/repos/{}/{}/commits",
            self.client.owner(), self.client.repo()
        );

        debug!("GitHub API URL: {}", url);
        debug!("Query parameters: path={}, per_page=1", encoded_path);
        debug!("Getting last modified time - Original path: {}, Encoded path: {}",
            file_path, encoded_path);

        let response = self.client.client()
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .query(&[("path", encoded_path.as_str()), ("per_page", "1")])
            .send()
            .await?;

        // Update rate limits from response headers
        self.update_rate_limits(response.headers()).await;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            error!("Failed to get last modified time. Status: {}, Error: {}", status, error_text);
            
            return match status.as_u16() {
                404 => Err(Box::new(GitHubError::NotFound(file_path.to_string()))),
                429 => {
                    let limits = self.rate_limits.read().await;
                    if let Some(info) = limits.get("core") {
                        Err(Box::new(GitHubError::RateLimitExceeded(info.clone())))
                    } else {
                        Err(format!("Rate limit exceeded without limit info").into())
                    }
                },
                _ => Err(format!("GitHub API error: {} - {}", status, error_text).into())
            };
        }

        let response_text = response.text().await?;
        debug!("GitHub API Response for commits: {}", response_text);
        
        let commits: Vec<serde_json::Value> = serde_json::from_str(&response_text)?;
        
        if commits.is_empty() {
            error!("Empty commits array returned for path: {} (encoded: {})", file_path, encoded_path);
            return Err(Box::new(GitHubError::NotFound(format!("No commit history found for {}", file_path))));
        }
        
        if let Some(last_commit) = commits.first() {
            debug!("Found commit data: {}", serde_json::to_string_pretty(last_commit)?);
            if let Some(commit) = last_commit["commit"]["committer"]["date"].as_str() {
                if let Ok(date) = DateTime::parse_from_rfc3339(commit) {
                    return Ok(date.with_timezone(&Utc));
                } else {
                    error!("Failed to parse commit date: {}", commit);
                    return Err("Failed to parse commit date from GitHub response".into());
                }
            } else {
                error!("No committer date found in commit data");
                return Err("No committer date found in GitHub response".into());
            }
        } else {
            error!("No commits found for file: {} (encoded path: {})", file_path, encoded_path);
            return Err(format!("No commit history found for file: {} (API path: {})", file_path, encoded_path).into());
        }
    }

    /// List all markdown files in a directory
    pub async fn list_markdown_files(&self, path: &str) -> Result<Vec<GitHubFileMetadata>, Box<dyn Error + Send + Sync>> {
        // Use GitHubClient's contents URL construction
        let url = self.client.get_contents_url(path);
        
        info!("GitHub API Request: URL={}, Original Path={}",
            url, path);

        let response = self.client.client()
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        let status = response.status();
        let headers = response.headers().clone();
        
        info!("GitHub API Response: Status={}, Headers={:?}", status, headers);

        let body = response.text().await?;
        info!("GitHub API Response Body (first 1000 chars): {}", &body[..body.len().min(1000)]);

        if !status.is_success() {
            let error_msg = match serde_json::from_str::<serde_json::Value>(&body) {
                Ok(error_json) => {
                    let msg = error_json["message"].as_str().unwrap_or("Unknown error");
                    format!("GitHub API error: {} - {}", status, msg)
                },
                Err(_) => format!("GitHub API error: {} - {}", status, body)
            };
            error!("{}", error_msg);
            return Err(error_msg.into());
        }

        let contents: Vec<serde_json::Value> = serde_json::from_str(&body)?;

        let settings = self.client.settings().read().await;
        let debug_enabled = settings.system.debug.enabled;
        drop(settings);
        
        let mut markdown_files = Vec::new();
        
        for item in contents {
            if item["type"].as_str().unwrap_or("") == "file" && 
               item["name"].as_str().unwrap_or("").ends_with(".md") {
                let name = item["name"].as_str().unwrap_or("").to_string();
                
                // In debug mode, only process Debug Test Page.md and debug linked node.md
                if debug_enabled && !name.contains("Debug Test Page") && !name.contains("debug linked node") {
                    continue;
                }

                debug!("Processing markdown file: {}", name);
                
                // Use the file name directly since base path is already handled
                debug!("Repository path for commits query: {}", name);
                
                let last_modified = match self.get_file_last_modified(&name).await {
                    Ok(time) => Some(time),
                    Err(e) => {
                        error!("Failed to get last modified time for {}: {}", name, e);
                        // Don't skip the file, just use current time as fallback
                        Some(Utc::now())
                    }
                };
                
                markdown_files.push(GitHubFileMetadata {
                    name,
                    sha: item["sha"].as_str().unwrap_or("").to_string(),
                    download_url: item["download_url"].as_str().unwrap_or("").to_string(),
                    etag: None,
                    last_checked: Some(Utc::now()),
                    last_modified,
                });
            }
        }

        if debug_enabled {
            info!("Debug mode: Processing only debug test files");
        }

        info!("Found {} markdown files", markdown_files.len());
        Ok(markdown_files)
    }
}