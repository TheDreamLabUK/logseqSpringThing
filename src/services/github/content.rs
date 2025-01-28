use super::api::GitHubClient;
use super::types::GitHubFileMetadata;
use chrono::{DateTime, Utc};
use log::{debug, error, info};
use std::error::Error;

/// Handles GitHub content API operations
#[derive(Clone)]
pub struct ContentAPI<'a> {
    client: &'a GitHubClient,
}

impl<'a> ContentAPI<'a> {
    /// Create a new ContentAPI instance
    pub fn new(client: &'a GitHubClient) -> Self {
        Self { client }
    }

    /// Check if a file is public by reading just the first line
    pub async fn check_file_public(&self, download_url: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let mut retries = 0;
        let (_, max_retries, retry_delay) = GitHubClient::constants();

        loop {
            match self.client.client()
                .get(download_url)
                .header("Authorization", format!("Bearer {}", self.client.token()))
                .header("Accept", "application/vnd.github+json")
                .header("Range", "bytes=0-12") // Exactly enough for "public:: true\n"
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() || status.as_u16() == 206 { // 206 Partial Content
                        match response.text().await {
                            Ok(content) => {
                                debug!("First line check ({}): '{}'", download_url, content.trim());
                                return Ok(content.trim() == "public:: true");
                            },
                            Err(e) => {
                                error!("Failed to read response content: {}", e);
                                if retries >= max_retries {
                                    return Err(e.into());
                                }
                            }
                        }
                    } else {
                        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        error!("Failed to check file public status. Status: {}, Error: {}", status, error_text);
                        
                        if status.as_u16() == 429 || (status.as_u16() >= 500 && status.as_u16() < 600) {
                            if retries >= max_retries {
                                return Err(format!("Failed after {} retries: {}", max_retries, error_text).into());
                            }
                        } else {
                            return Err(format!("GitHub API error: {} - {}", status, error_text).into());
                        }
                    }
                }
                Err(e) => {
                    error!("Request failed: {}", e);
                    if retries >= max_retries {
                        return Err(e.into());
                    }
                }
            }

            retries += 1;
            info!("Retrying public check request ({}/{})", retries, max_retries);
            tokio::time::sleep(retry_delay).await;
        }
    }

    /// Fetch full content of a file
    pub async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut retries = 0;
        let (_, max_retries, retry_delay) = GitHubClient::constants();

        loop {
            match self.client.client()
                .get(download_url)
                .header("Authorization", format!("Bearer {}", self.client.token()))
                .header("Accept", "application/vnd.github+json")
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        match response.text().await {
                            Ok(content) => return Ok(content),
                            Err(e) => {
                                error!("Failed to read response content: {}", e);
                                if retries >= max_retries {
                                    return Err(e.into());
                                }
                            }
                        }
                    } else {
                        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        error!("Failed to fetch file content. Status: {}, Error: {}", status, error_text);
                        
                        if status.as_u16() == 429 || (status.as_u16() >= 500 && status.as_u16() < 600) {
                            if retries >= max_retries {
                                return Err(format!("Failed after {} retries: {}", max_retries, error_text).into());
                            }
                        } else {
                            return Err(format!("GitHub API error: {} - {}", status, error_text).into());
                        }
                    }
                }
                Err(e) => {
                    error!("Request failed: {}", e);
                    if retries >= max_retries {
                        return Err(e.into());
                    }
                }
            }

            retries += 1;
            info!("Retrying request ({}/{})", retries, max_retries);
            tokio::time::sleep(retry_delay).await;
        }
    }

    /// Get the last modified time for a file
    pub async fn get_file_last_modified(&self, file_path: &str) -> Result<DateTime<Utc>, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/commits",
            self.client.owner(), self.client.repo()
        );

        let encoded_path = url::form_urlencoded::byte_serialize(file_path.as_bytes()).collect::<String>();
        
        debug!("Getting last modified time - Original path: {}, Encoded path: {}",
            file_path, encoded_path);
        
        let response = self.client.client()
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .query(&[("path", encoded_path.as_str()), ("per_page", "1")])
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            error!("Failed to get last modified time. Status: {}, Error: {}", status, error_text);
            return Err(format!("GitHub API error: {} - {}", status, error_text).into());
        }

        let commits: Vec<serde_json::Value> = response.json().await?;
        
        if let Some(last_commit) = commits.first() {
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
            error!("No commits found for file: {}", file_path);
            return Err(format!("No commit history found for file: {}", file_path).into());
        }
    }

    /// List all markdown files in a directory
    pub async fn list_markdown_files(&self, path: &str) -> Result<Vec<GitHubFileMetadata>, Box<dyn Error + Send + Sync>> {
        let encoded_path = self.client.get_api_path();
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.client.owner(),
            self.client.repo(),
            encoded_path
        );
        
        info!("GitHub API Request: URL={}, Encoded Path={}, Original Path={}",
            url, encoded_path, path);

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
                
                let last_modified = match self.get_file_last_modified(&self.client.get_full_path(&name)).await {
                    Ok(time) => Some(time),
                    Err(e) => {
                        error!("Failed to get last modified time for {}: {}", name, e);
                        continue;
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