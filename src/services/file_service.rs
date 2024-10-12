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
}

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn std::error::Error + Send + Sync>>;
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>>;
    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
}

impl RealGitHubService {
    pub async fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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

    async fn fetch_directory_contents(&self, path: &str) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error + Send + Sync>> {
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

    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let content = self.client.get(download_url)
            .header("Authorization", format!("token {}", self.token))
            .header("User-Agent", "rust-github-api")
            .send()
            .await?
            .text()
            .await?;
        Ok(content)
    }

    /// Fetches the metadata of Markdown files from the specified GitHub directory.
    pub async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn std::error::Error + Send + Sync>> {
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

    /// Retrieves the download URL for a specific file.
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
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
    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        self.fetch_file_metadata().await
    }

    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
        self.get_download_url(file_name).await
    }

    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.fetch_file_content(download_url).await
    }
}

pub struct FileService;

impl FileService {
    pub async fn fetch_and_process_files(
        github_service: &dyn GitHubService,
        settings: Arc<RwLock<Settings>>,
        metadata_map: &mut HashMap<String, Metadata>,
    ) -> Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>> {
        let github_files_metadata = github_service.fetch_file_metadata().await?;
        debug!("Fetched {} file metadata from GitHub", github_files_metadata.len());

        let mut processed_files = Vec::new();
        
        // Clone local metadata for comparison
        let local_metadata = metadata_map.clone();
        
        // Create a HashSet of GitHub file names for removal detection
        let github_file_names: HashSet<String> = github_files_metadata.iter().map(|f| f.name.clone()).collect();
        
        // Handle removed files
        let removed_files: Vec<String> = local_metadata.keys()
            .filter(|name| !github_file_names.contains(*name))
            .cloned()
            .collect();
        
        for removed_file in removed_files {
            log::info!("Removing file not present on GitHub: {}", removed_file);
            metadata_map.remove(&removed_file);
            // Optionally, remove the local file from the filesystem
            // fs::remove_file(format!("/app/data/markdown/{}", removed_file))?;
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

            // Fetch file content since it's new or updated
                if let Some(download_url) = github_service.get_download_url(&file_meta.name).await? {
                    let content = github_service.fetch_file_content(&download_url).await?;
                    
                    // Check if the first line is "public:: true"
                    let first_line = content.lines().next().unwrap_or("").trim();
                    if first_line != "public:: true" {
                        debug!("File '{}' is not public. Skipping.", file_meta.name);
                        continue;
                    }

                    // Update local file and metadata
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
                    });
                    debug!("Processed and updated file: {}", file_meta.name);
                } else {
                    log::error!("Download URL not found for file: {}", file_meta.name);
            }
        }

        debug!("Processed {} files after comparison", processed_files.len());
        Ok(processed_files)
    }

    fn count_topics(content: &str, metadata_map: &HashMap<String, Metadata>) -> HashMap<String, usize> {
        metadata_map.keys()
            .filter_map(|file_name| {
                let topic = file_name.trim_end_matches(".md");
                let count = content.matches(topic).count();
                if count > 0 {
                    Some((topic.to_string(), count))
                } else {
                    None
                }
            })
            .collect()
    }

    fn save_metadata(metadata_map: &HashMap<String, Metadata>) -> Result<(), std::io::Error> {
        let metadata_path = "/app/data/markdown/metadata.json";
        let updated_content = serde_json::to_string_pretty(metadata_map)?;
        fs::write(metadata_path, updated_content)?;
        debug!("Updated metadata file at: {}", metadata_path);
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

    fn load_local_metadata() -> Result<HashMap<String, Metadata>, Box<dyn std::error::Error + Send + Sync>> {
        let metadata_path = "/app/data/markdown/metadata.json";

        if Path::new(metadata_path).exists() {
            let metadata_content = fs::read_to_string(metadata_path)?;
            let metadata: HashMap<String, Metadata> = serde_json::from_str(&metadata_content)?;
            Ok(metadata)
        } else {
            Ok(HashMap::new())
        }
    }

    fn strip_double_brackets(content: &str) -> String {
        let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
        re.replace_all(content, "$1").to_string()
    }

    fn count_hyperlinks(content: &str) -> usize {
        let re = Regex::new(r"\[.*?\]\(.*?\)").unwrap();
        re.find_iter(content).count()
    }

    fn process_against_topics(content: &str, metadata_map: &HashMap<String, Metadata>) -> String {
        let mut processed_content = content.to_string();
        for topic in metadata_map.keys() {
            let topic_name = topic.trim_end_matches(".md");
            if content.contains(topic_name) {
                processed_content.push_str(&format!("\nRelated to topic: {}", topic_name));
            }
        }
        processed_content
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
        let metadata_path = "/app/data/markdown/metadata.json";

        let mut metadata_map = if Path::new(metadata_path).exists() {
            let content = fs::read_to_string(metadata_path)?;
            serde_json::from_str::<HashMap<String, Metadata>>(&content)?
        } else {
            HashMap::new()
        };

        metadata_map.insert(metadata.file_name.clone(), metadata.clone());

        let updated_content = serde_json::to_string_pretty(&metadata_map)?;

        fs::write(metadata_path, updated_content)?;
        debug!("Updated metadata file at: {}", metadata_path);

        Ok(())
    }
}