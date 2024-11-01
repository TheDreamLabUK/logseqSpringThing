use crate::config::GithubSettings;
use crate::models::metadata::Metadata;
use crate::config::Settings;
use log::{debug, info, error};
use reqwest::Client;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::collections::HashMap;
use base64;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use regex::Regex;
use sha1::{Sha1, Digest};
use std::collections::HashSet;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

const METADATA_PATH: &str = "/app/data/markdown/metadata.json";

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub is_file: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GithubFile {
    pub name: String,
    pub content: String,
    pub sha: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GithubFileMetadata {
    pub name: String,
    pub sha: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedFile {
    pub file_name: String,
    pub content: String,
    pub is_public: bool,
    pub metadata: Metadata,
}

#[derive(Debug)]
pub enum GitHubError {
    MissingCredentials(String),
    ApiError(String),
    NetworkError(String),
    Other(Box<dyn Error + Send + Sync>),
}

impl std::fmt::Display for GitHubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitHubError::MissingCredentials(msg) => write!(f, "GitHub credentials error: {}", msg),
            GitHubError::ApiError(msg) => write!(f, "GitHub API error: {}", msg),
            GitHubError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            GitHubError::Other(e) => write!(f, "Other error: {}", e),
        }
    }
}

impl Error for GitHubError {}

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn get_file_content(&self, path: &str) -> Result<String, Box<dyn Error + Send + Sync>>;
    async fn save_file(&self, path: &str, content: &str) -> Result<(), Box<dyn Error + Send + Sync>>;
    async fn list_files(&self, path: &str) -> Result<Vec<FileMetadata>, Box<dyn Error + Send + Sync>>;
    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn Error + Send + Sync>>;
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn Error + Send + Sync>>;
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
}

impl RealGitHubService {
    pub fn new(github_settings: &GithubSettings) -> Self {
        // Log detailed information about GitHub settings (excluding sensitive data)
        info!("Initializing GitHub service with owner: {}, repo: {}, directory: {}", 
            github_settings.github_owner,
            github_settings.github_repo,
            github_settings.github_directory
        );

        if github_settings.github_access_token.is_empty() {
            error!("GitHub access token is not configured");
        } else {
            debug!("GitHub access token starts with: {}", github_settings.github_access_token.chars().take(10).collect::<String>());
        }
        if github_settings.github_owner.is_empty() {
            error!("GitHub owner is not configured");
        }
        if github_settings.github_repo.is_empty() {
            error!("GitHub repository is not configured");
        }

        RealGitHubService {
            client: Client::new(),
            token: github_settings.github_access_token.clone(),
            owner: github_settings.github_owner.clone(),
            repo: github_settings.github_repo.clone(),
            base_path: github_settings.github_directory.clone(),
        }
    }

    fn validate_credentials(&self) -> Result<(), GitHubError> {
        if self.token.is_empty() {
            return Err(GitHubError::MissingCredentials("GitHub access token is not configured".into()));
        }
        if self.owner.is_empty() {
            return Err(GitHubError::MissingCredentials("GitHub owner is not configured".into()));
        }
        if self.repo.is_empty() {
            return Err(GitHubError::MissingCredentials("GitHub repository is not configured".into()));
        }
        Ok(())
    }

    async fn get_api_response(&self, path: &str) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>> {
        self.validate_credentials()?;

        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner, self.repo, path
        );

        debug!("Making GitHub API request to: {}", url);
        debug!("Using GitHub token starting with: {}", self.token.chars().take(10).collect::<String>());

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("User-Agent", "rust-github-api")
            .header("Accept", "application/vnd.github.v3+json")
            .send()
            .await
            .map_err(|e| GitHubError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_message = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!("GitHub API error: Status {}, Message: {}", status, error_message);
            return Err(Box::new(GitHubError::ApiError(format!("GitHub API returned error {}: {}", status, error_message))));
        }

        Ok(response)
    }
}

#[async_trait]
impl GitHubService for RealGitHubService {
    async fn get_file_content(&self, path: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let full_path = format!("{}/{}", self.base_path, path);
        let response = self.get_api_response(&full_path).await?;
        let content = response.text().await?;
        Ok(content)
    }

    async fn save_file(&self, path: &str, content: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.validate_credentials()?;

        let full_path = format!("{}/{}", self.base_path, path);
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner, self.repo, full_path
        );

        let response = self.client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("User-Agent", "rust-github-api")
            .header("Accept", "application/vnd.github.v3+json")
            .json(&serde_json::json!({
                "message": "Update file",
                "content": base64::encode(content),
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_message = response.text().await?;
            return Err(Box::new(GitHubError::ApiError(format!("Failed to save file: {}", error_message))));
        }

        Ok(())
    }

    async fn list_files(&self, path: &str) -> Result<Vec<FileMetadata>, Box<dyn Error + Send + Sync>> {
        let full_path = format!("{}/{}", self.base_path, path);
        let response = self.get_api_response(&full_path).await?;
        let items: Vec<serde_json::Value> = response.json().await?;
        
        let files = items.into_iter()
            .map(|item| FileMetadata {
                name: item["name"].as_str().unwrap_or("").to_string(),
                path: item["path"].as_str().unwrap_or("").to_string(),
                size: item["size"].as_u64().unwrap_or(0),
                is_file: item["type"].as_str().unwrap_or("") == "file",
            })
            .collect();
        
        Ok(files)
    }

    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn Error + Send + Sync>> {
        let response = self.get_api_response(&self.base_path).await?;
        let items: Vec<serde_json::Value> = response.json().await?;
        
        let metadata = items.into_iter()
            .filter(|item| item["type"].as_str().unwrap_or("") == "file")
            .map(|item| GithubFileMetadata {
                name: item["name"].as_str().unwrap_or("").to_string(),
                sha: item["sha"].as_str().unwrap_or("").to_string(),
            })
            .collect();
        
        Ok(metadata)
    }

    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
        let full_path = format!("{}/{}", self.base_path, file_name);
        let response = self.get_api_response(&full_path).await?;
        let item: serde_json::Value = response.json().await?;
        Ok(item["download_url"].as_str().map(|s| s.to_string()))
    }
}

pub struct FileService;

impl FileService {
    pub async fn fetch_and_process_files(
        github_service: &dyn GitHubService,
        settings: Arc<RwLock<Settings>>,
        metadata_map: &mut HashMap<String, Metadata>,
    ) -> Result<Vec<ProcessedFile>, Box<dyn Error + Send + Sync>> {
        let github_files_metadata = github_service.fetch_file_metadata().await?;
        debug!("Fetched {} file metadata from GitHub", github_files_metadata.len());

        let mut processed_files = Vec::new();
        let local_metadata = metadata_map.clone();
        let github_file_names: HashSet<String> = github_files_metadata.iter().map(|f| f.name.clone()).collect();
        
        let removed_files: Vec<String> = local_metadata.keys()
            .filter(|name| !github_file_names.contains(*name))
            .cloned()
            .collect();
        
        for removed_file in removed_files {
            info!("Removing file not present on GitHub: {}", removed_file);
            metadata_map.remove(&removed_file);
        }

        for file_meta in github_files_metadata {
            let local_meta = metadata_map.get(&file_meta.name);
            if let Some(local_meta) = local_meta {
                if local_meta.sha1 == file_meta.sha {
                    debug!("File '{}' is up-to-date. Skipping.", file_meta.name);
                    continue;
                } else {
                    info!("File '{}' has been updated. Fetching new content.", file_meta.name);
                }
            } else {
                info!("New file detected: '{}'. Fetching content.", file_meta.name);
            }

            if let Some(download_url) = github_service.get_download_url(&file_meta.name).await? {
                let content = github_service.get_file_content(&download_url).await?;
                
                let first_line = content.lines().next().unwrap_or("").trim();
                let is_public = first_line == "public:: true";

                fs::write(format!("/app/data/markdown/{}", file_meta.name), &content)?;
                let new_metadata = Metadata {
                    file_name: file_meta.name.clone(),
                    file_size: content.len(),
                    hyperlink_count: Self::count_hyperlinks(&content),
                    sha1: file_meta.sha.clone(),
                    last_modified: Utc::now(),
                    perplexity_link: String::new(),
                    last_perplexity_process: None,
                    topic_counts: HashMap::new(),
                };
                metadata_map.insert(file_meta.name.clone(), new_metadata.clone());
                
                processed_files.push(ProcessedFile {
                    file_name: file_meta.name.clone(),
                    content,
                    is_public,
                    metadata: new_metadata,
                });
                debug!("Processed and updated file: {}", file_meta.name);
            } else {
                info!("Download URL not found for file: {}", file_meta.name);
            }
        }

        debug!("Processed {} files after comparison", processed_files.len());
        Ok(processed_files)
    }

    pub fn load_or_create_metadata() -> Result<HashMap<String, Metadata>, Box<dyn Error + Send + Sync>> {
        if Path::new(METADATA_PATH).exists() {
            let metadata_content = fs::read_to_string(METADATA_PATH)?;
            let metadata: HashMap<String, Metadata> = serde_json::from_str(&metadata_content)?;
            Ok(metadata)
        } else {
            debug!("metadata.json not found. Creating a new one.");
            let empty_metadata = HashMap::new();
            Self::save_metadata(&empty_metadata)?;
            Ok(empty_metadata)
        }
    }

    pub fn save_metadata(metadata_map: &HashMap<String, Metadata>) -> Result<(), std::io::Error> {
        let metadata_path = METADATA_PATH;
        
        if let Some(parent_dir) = Path::new(metadata_path).parent() {
            fs::create_dir_all(parent_dir)?;
            debug!("Ensured directory exists: {}", parent_dir.display());
        }

        let updated_content = serde_json::to_string_pretty(metadata_map)?;
        fs::write(metadata_path, updated_content)?;
        debug!("Updated metadata file at: {}", metadata_path);
        Ok(())
    }

    pub fn update_metadata(metadata_map: &HashMap<String, Metadata>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let existing_metadata = Self::load_or_create_metadata()?;
        let mut updated_metadata = existing_metadata;

        for (key, value) in metadata_map {
            updated_metadata.insert(key.clone(), value.clone());
        }

        Self::save_metadata(&updated_metadata)?;
        info!("Updated metadata.json file at {}", METADATA_PATH);

        Ok(())
    }

    fn count_hyperlinks(content: &str) -> usize {
        let re = Regex::new(r"\[.*?\]\(.*?\)").unwrap();
        re.find_iter(content).count()
    }

    pub fn save_file_metadata(metadata: Metadata) -> Result<(), std::io::Error> {
        info!("Saving metadata for file: {}", metadata.file_name);

        let markdown_path = format!("/app/data/markdown/{}", metadata.file_name);

        if let Some(parent) = Path::new(&markdown_path).parent() {
            fs::create_dir_all(parent)?;
            debug!("Ensured directory exists: {}", parent.display());
        }

        fs::write(&markdown_path, &metadata.perplexity_link)?;
        debug!("Written processed content to: {}", markdown_path);

        Self::update_metadata_file(&metadata)?;

        Ok(())
    }

    fn update_metadata_file(metadata: &Metadata) -> Result<(), std::io::Error> {
        if let Some(parent_dir) = Path::new(METADATA_PATH).parent() {
            fs::create_dir_all(parent_dir)?;
            debug!("Ensured directory exists: {}", parent_dir.display());
        }

        let mut metadata_map = if Path::new(METADATA_PATH).exists() {
            let content = fs::read_to_string(METADATA_PATH)?;
            serde_json::from_str::<HashMap<String, Metadata>>(&content)?
        } else {
            HashMap::new()
        };

        metadata_map.insert(metadata.file_name.clone(), metadata.clone());

        let updated_content = serde_json::to_string_pretty(&metadata_map)?;
        fs::write(METADATA_PATH, updated_content)?;
        debug!("Updated metadata file at: {}", METADATA_PATH);

        Ok(())
    }
}
