use crate::models::metadata::Metadata;
use crate::config::Settings;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use async_trait::async_trait;
use log::{info, debug, error};
use regex::Regex;
use sha1::{Sha1, Digest};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::error::Error as StdError;

const METADATA_PATH: &str = "/app/data/markdown/metadata.json";

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

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>>;
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>>;
    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>>;
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
}

impl RealGitHubService {
    pub async fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let settings = settings.read().await;
        let github_settings = &settings.github;
        if github_settings.github_access_token.is_empty() {
            return Err("GitHub access token is empty".into());
        }
        Ok(Self {
            client: Client::new(),
            token: github_settings.github_access_token.clone(),
            owner: github_settings.github_owner.clone(),
            repo: github_settings.github_repo.clone(),
            base_path: github_settings.github_directory.clone(),
        })
    }

    async fn fetch_directory_contents(&self, path: &str) -> Result<Vec<serde_json::Value>, Box<dyn StdError + Send + Sync>> {
        let url = format!("https://api.github.com/repos/{}/{}/contents/{}", self.owner, self.repo, path);
        debug!("Fetching contents from GitHub: {}", url);

        let response = self.client.get(&url)
            .header("Authorization", format!("token {}", self.token))
            .header("User-Agent", "rust-github-api")
            .send()
            .await?;

        debug!("GitHub API response status: {}", response.status());

        let response_body = response.text().await?;
        debug!("GitHub API response body: {}", response_body);

        let contents: Vec<serde_json::Value> = serde_json::from_str(&response_body)
            .map_err(|e| {
                error!("Failed to parse GitHub API response: {}", e);
                format!("Failed to parse GitHub API response: {}. Response body: {}", e, response_body)
            })?;

        // Filter out any subdirectories
        let files_only = contents.into_iter()
            .filter(|item| item["type"].as_str() == Some("file"))
            .collect();

        Ok(files_only)
    }

    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>> {
        let content = self.client.get(download_url)
            .header("Authorization", format!("token {}", self.token))
            .header("User-Agent", "rust-github-api")
            .send()
            .await?
            .text()
            .await?;
        Ok(content)
    }

    pub async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        let mut github_files_metadata = Vec::new();
        
        let contents = self.fetch_directory_contents(&self.base_path).await?;
    
        for item in contents {
            let name = item["name"].as_str().unwrap_or("");
            let item_type = item["type"].as_str().unwrap_or("");
    
            if item_type == "file" && name.ends_with(".md") {
                let sha = item["sha"].as_str().unwrap_or("").to_string();
                github_files_metadata.push(GithubFileMetadata {
                    name: name.to_string(),
                    sha,
                });
            }
        }
    
        debug!("Fetched metadata for {} Markdown files from GitHub", github_files_metadata.len());
        Ok(github_files_metadata)
    }

    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>> {
        let url = format!("https://api.github.com/repos/{}/{}/contents/{}/{}", 
            self.owner, self.repo, self.base_path, file_name);
        debug!("Fetching download URL for file: {}", url);

        let response = self.client.get(&url)
            .header("Authorization", format!("token {}", self.token))
            .header("User-Agent", "rust-github-api")
            .send()
            .await?;

        if response.status().is_success() {
            let response_body = response.text().await?;
            let json: serde_json::Value = serde_json::from_str(&response_body)?;
            Ok(json["download_url"].as_str().map(|s| s.to_string()))
        } else {
            error!("Failed to fetch download URL: {}", response.status());
            Ok(None)
        }
    }
}

#[async_trait]
impl GitHubService for RealGitHubService {
    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        self.fetch_file_metadata().await
    }

    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>> {
        self.get_download_url(file_name).await
    }

    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>> {
        self.fetch_file_content(download_url).await
    }
}

pub struct FileService;

impl FileService {
    pub async fn fetch_and_process_files(
        github_service: &dyn GitHubService,
        _settings: Arc<RwLock<Settings>>,
        metadata_map: &mut HashMap<String, Metadata>,
    ) -> Result<Vec<ProcessedFile>, Box<dyn StdError + Send + Sync>> {
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
            log::info!("Removing file not present on GitHub: {}", removed_file);
            metadata_map.remove(&removed_file);
        }

        for file_meta in github_files_metadata {
            let local_meta = metadata_map.get(&file_meta.name);
            if let Some(local_meta) = local_meta {
                if local_meta.sha1 == file_meta.sha {
                    debug!("File '{}' is up-to-date. Skipping.", file_meta.name);
                    continue;
                } else {
                    log::info!("File '{}' has been updated. Fetching new content.", file_meta.name);
                }
            } else {
                log::info!("New file detected: '{}'. Fetching content.", file_meta.name);
            }

            if let Some(download_url) = github_service.get_download_url(&file_meta.name).await? {
                let content = github_service.fetch_file_content(&download_url).await?;
                
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
                log::error!("Download URL not found for file: {}", file_meta.name);
            }
        }

        debug!("Processed {} files after comparison", processed_files.len());
        Ok(processed_files)
    }

    pub fn load_or_create_metadata() -> Result<HashMap<String, Metadata>, Box<dyn StdError + Send + Sync>> {
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

    pub fn update_metadata(metadata_map: &HashMap<String, Metadata>) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let existing_metadata = Self::load_or_create_metadata()?;
        let mut updated_metadata = existing_metadata;

        for (key, value) in metadata_map {
            updated_metadata.insert(key.clone(), value.clone());
        }

        Self::save_metadata(&updated_metadata)?;
        info!("Updated metadata.json file at {}", METADATA_PATH);

        Ok(())
    }

    fn should_process_file(file: &GithubFile) -> bool {
        let first_line = file.content.lines().next().unwrap_or("").trim();
        first_line == "public:: true"
    }

    fn calculate_sha1(content: &str) -> String {
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
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
