use crate::models::metadata::{Metadata, MetadataStore, MetadataOps};
use crate::models::graph::GraphData;
use crate::config::Settings;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use async_trait::async_trait;
use log::{info, debug, error};
use regex::Regex;
use std::fs;
use std::path::Path;
use chrono::{Utc, DateTime};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::error::Error as StdError;
use std::time::Duration;
use tokio::time::sleep;
use actix_web::web;
use std::collections::HashMap;
use std::fs::File;
use std::io::Error;

// Constants
const METADATA_PATH: &str = "/app/data/metadata/metadata.json";
pub const MARKDOWN_DIR: &str = "/app/data/markdown";
const GITHUB_API_DELAY: Duration = Duration::from_millis(500); // Increased delay for GitHub rate limits
const MAX_RETRIES: u32 = 3; // Maximum number of retries for API calls
const RETRY_DELAY: Duration = Duration::from_secs(2); // Delay between retries
const MIN_SIZE: f64 = 5.0;  // Minimum node size
const MAX_SIZE: f64 = 50.0; // Maximum node size

#[derive(Serialize, Deserialize, Clone)]
pub struct GithubFile {
    pub name: String,
    pub path: String,
    pub sha: String,
    pub size: usize,
    pub url: String,
    pub download_url: String,
}

#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct GithubFileMetadata {
    pub name: String,
    pub sha: String,
    pub download_url: String,
    pub etag: Option<String>,
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub last_checked: Option<DateTime<Utc>>,
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub last_modified: Option<DateTime<Utc>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedFile {
    pub file_name: String,
    pub content: String,
    pub is_public: bool,
    pub metadata: Metadata,
}

// TODO: This struct will be used in future implementation of reference tracking
#[allow(dead_code)]
struct ReferenceInfo {
    file_name: String,
    references: Vec<String>,
}

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn fetch_file_metadata(&self, skip_debug_filter: bool) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>>;
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>>;
    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>>;
    async fn get_file_last_modified(&self, file_path: &str) -> Result<DateTime<Utc>, Box<dyn StdError + Send + Sync>>;
    async fn fetch_files(&self, path: &str) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>>;
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
    settings: Arc<RwLock<Settings>>,
}

impl RealGitHubService {
    pub fn new(
        token: String,
        owner: String,
        repo: String,
        base_path: String,
        settings: Arc<RwLock<Settings>>,
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let client = Client::builder()
            .user_agent("rust-github-api")
            .timeout(Duration::from_secs(30))
            .build()?;

        // Trim any leading/trailing slashes from base_path
        let base_path = base_path.trim_matches('/').to_string();

        debug!("Initializing GitHub service with base_path: {}", base_path);

        Ok(Self {
            client,
            token,
            owner,
            repo,
            base_path,
            settings: Arc::clone(&settings),
        })
    }

    fn get_full_path(&self, path: &str) -> String {
        let base = self.base_path.trim_matches('/');
        let path = path.trim_matches('/');
        
        // Always include base path if it exists
        if !base.is_empty() {
            if path.is_empty() {
                base.to_string()
            } else {
                format!("{}/{}", base, path)
            }
        } else {
            path.to_string()
        }
    }

    fn get_api_path(&self) -> String {
        // For API requests, always include the base path
        if !self.base_path.is_empty() {
            self.base_path.trim_matches('/').to_string()
        } else {
            String::new()
        }
    }
}

#[async_trait]
impl GitHubService for RealGitHubService {
    async fn fetch_files(&self, _path: &str) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        self.fetch_file_metadata(false).await
    }

    async fn fetch_file_metadata(&self, skip_debug_filter: bool) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner,
            self.repo,
            self.get_api_path()
        );
        
        info!("GitHub API Request: URL={}, Token={}, Owner={}, Repo={}, BasePath={}", 
            url, 
            self.token.chars().take(4).collect::<String>() + "...", 
            self.owner,
            self.repo,
            self.base_path
        );

        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
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

        let contents: Vec<serde_json::Value> = match serde_json::from_str(&body) {
            Ok(parsed) => parsed,
            Err(e) => {
                error!("Failed to parse GitHub API response: {}", e);
                error!("Response body: {}", body);
                return Err(Box::new(e));
            }
        };

        let settings = self.settings.read().await;
        let debug_enabled = settings.server_debug.enabled;
        drop(settings);
        
        let mut markdown_files = Vec::new();
        
        for item in contents {
            if item["type"].as_str().unwrap_or("") == "file" && 
               item["name"].as_str().unwrap_or("").ends_with(".md") {
                let name = item["name"].as_str().unwrap_or("").to_string();
                
                // In debug mode and not skipping filter, only process Debug Test Page.md and debug linked node.md
                if !skip_debug_filter && debug_enabled && !name.contains("Debug Test Page") && !name.contains("debug linked node") {
                    continue;
                }

                debug!("Processing markdown file: {}", name);
                
                let last_modified = match self.get_file_last_modified(&self.get_full_path(&name)).await {
                    Ok(time) => Some(time),
                    Err(e) => {
                        error!("Failed to get last modified time for {}: {}", name, e);
                        continue;
                    }
                };
                
                markdown_files.push(GithubFileMetadata {
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

    async fn get_download_url(&self, _file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner,
            self.repo,
            self.get_api_path()
        );

        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        if response.status().is_success() {
            let file: GithubFile = response.json().await?;
            Ok(Some(file.download_url))
        } else {
            Ok(None)
        }
    }

    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>> {
        let mut retries = 0;
        loop {
            match self.client.get(download_url)
                .header("Authorization", format!("Bearer {}", self.token))
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
                                if retries >= MAX_RETRIES {
                                    return Err(e.into());
                                }
                            }
                        }
                    } else {
                        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        error!("Failed to fetch file content. Status: {}, Error: {}", status, error_text);
                        
                        // Check if we should retry based on status code
                        if status.as_u16() == 429 || (status.as_u16() >= 500 && status.as_u16() < 600) {
                            if retries >= MAX_RETRIES {
                                return Err(format!("Failed after {} retries: {}", MAX_RETRIES, error_text).into());
                            }
                        } else {
                            return Err(format!("GitHub API error: {} - {}", status, error_text).into());
                        }
                    }
                }
                Err(e) => {
                    error!("Request failed: {}", e);
                    if retries >= MAX_RETRIES {
                        return Err(e.into());
                    }
                }
            }

            retries += 1;
            info!("Retrying request ({}/{})", retries, MAX_RETRIES);
            sleep(RETRY_DELAY).await;
        }
    }

    async fn get_file_last_modified(&self, file_path: &str) -> Result<DateTime<Utc>, Box<dyn StdError + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/commits",
            self.owner, self.repo
        );

        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .query(&[("path", file_path), ("per_page", "1")])
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
}

pub struct FileService {
    #[allow(dead_code)]
    settings: Arc<RwLock<Settings>>,
}

impl FileService {
    pub fn new(settings: Arc<RwLock<Settings>>) -> Self {
        Self {
            settings
        }
    }

    /// Process uploaded file and return graph data
    pub async fn process_file_upload(&self, payload: web::Bytes) -> Result<GraphData, Error> {
        let content = String::from_utf8(payload.to_vec())
            .map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let metadata = Self::load_or_create_metadata()
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;
        let mut graph_data = GraphData::new();
        
        // Create a temporary file to process
        let temp_filename = format!("temp_{}.md", Utc::now().timestamp());
        let temp_path = format!("{}/{}", MARKDOWN_DIR, temp_filename);
        if let Err(e) = fs::write(&temp_path, &content) {
            return Err(Error::new(std::io::ErrorKind::Other, e.to_string()));
        }

        // Extract references and create metadata
        let valid_nodes: Vec<String> = metadata.keys()
            .map(|name| name.trim_end_matches(".md").to_string())
            .collect();

        let references = Self::extract_references(&content, &valid_nodes);
        let topic_counts = Self::convert_references_to_topic_counts(references);

        // Create metadata for the uploaded file
        let file_size = content.len();
        let node_size = Self::calculate_node_size(file_size);
        let file_metadata = Metadata {
            file_name: temp_filename.clone(),
            file_size,
            node_size,
            hyperlink_count: Self::count_hyperlinks(&content),
            sha1: Self::calculate_sha1(&content),
            last_modified: Utc::now(),
            perplexity_link: String::new(),
            last_perplexity_process: None,
            topic_counts,
        };

        // Update graph data
        graph_data.metadata.insert(temp_filename.clone(), file_metadata);

        // Clean up temporary file
        if let Err(e) = fs::remove_file(&temp_path) {
            error!("Failed to remove temporary file: {}", e);
        }

        Ok(graph_data)
    }

    /// List available files
    pub async fn list_files(&self) -> Result<Vec<String>, Error> {
        let metadata = Self::load_or_create_metadata()
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;
        Ok(metadata.keys().cloned().collect())
    }

    /// Load a specific file and return graph data
    pub async fn load_file(&self, filename: &str) -> Result<GraphData, Error> {
        let file_path = format!("{}/{}", MARKDOWN_DIR, filename);
        if !Path::new(&file_path).exists() {
            return Err(Error::new(std::io::ErrorKind::NotFound, format!("File not found: {}", filename)));
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let metadata = Self::load_or_create_metadata()
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;
        let mut graph_data = GraphData::new();

        // Extract references and update metadata
        let valid_nodes: Vec<String> = metadata.keys()
            .map(|name| name.trim_end_matches(".md").to_string())
            .collect();

        let references = Self::extract_references(&content, &valid_nodes);
        let topic_counts = Self::convert_references_to_topic_counts(references);

        // Update or create metadata for the file
        let file_size = content.len();
        let node_size = Self::calculate_node_size(file_size);
        let file_metadata = Metadata {
            file_name: filename.to_string(),
            file_size,
            node_size,
            hyperlink_count: Self::count_hyperlinks(&content),
            sha1: Self::calculate_sha1(&content),
            last_modified: Utc::now(),
            perplexity_link: String::new(),
            last_perplexity_process: None,
            topic_counts,
        };

        // Update graph data
        graph_data.metadata.insert(filename.to_string(), file_metadata);
        
        Ok(graph_data)
    }

    /// Load metadata from file or create new if not exists
    pub fn load_or_create_metadata() -> Result<MetadataStore, String> {
        // Ensure metadata directory exists
        std::fs::create_dir_all("/app/data/metadata")
            .map_err(|e| format!("Failed to create metadata directory: {}", e))?;
        
        let metadata_path = "/app/data/metadata/metadata.json";
        
        if let Ok(file) = File::open(metadata_path) {
            info!("Loading existing metadata from {}", metadata_path);
            serde_json::from_reader(file)
                .map_err(|e| format!("Failed to parse metadata: {}", e))
        } else {
            info!("Creating new metadata file at {}", metadata_path);
            let empty_store = MetadataStore::default();
            let file = File::create(metadata_path)
                .map_err(|e| format!("Failed to create metadata file: {}", e))?;
                
            serde_json::to_writer_pretty(file, &empty_store)
                .map_err(|e| format!("Failed to write metadata: {}", e))?;
                
            // Verify file was created with correct permissions
            let metadata = std::fs::metadata(metadata_path)
                .map_err(|e| format!("Failed to verify metadata file: {}", e))?;
            
            if !metadata.is_file() {
                return Err("Metadata file was not created properly".to_string());
            }
            
            Ok(empty_store)
        }
    }

    /// Calculate node size based on file size
    fn calculate_node_size(file_size: usize) -> f64 {
        const BASE_SIZE: f64 = 1000.0; // Base file size for scaling

        let size = (file_size as f64 / BASE_SIZE).min(5.0);
        MIN_SIZE + (size * (MAX_SIZE - MIN_SIZE) / 5.0)
    }

    /// Extract references to other files based on their names (case insensitive)
    fn extract_references(content: &str, valid_nodes: &[String]) -> Vec<String> {
        let mut references = Vec::new();
        let content_lower = content.to_lowercase();
        
        for node_name in valid_nodes {
            let node_name_lower = node_name.to_lowercase();
            
            // Create a regex pattern with word boundaries
            let pattern = format!(r"\b{}\b", regex::escape(&node_name_lower));
            if let Ok(re) = Regex::new(&pattern) {
                // Count case-insensitive matches of the filename
                let count = re.find_iter(&content_lower).count();
                
                // If we found any references, add them to the map
                if count > 0 {
                    debug!("Found {} references to {} in content", count, node_name);
                    // Add the reference multiple times based on count
                    for _ in 0..count {
                        references.push(node_name.clone());
                    }
                }
            }
        }
        
        references
    }

    fn convert_references_to_topic_counts(references: Vec<String>) -> HashMap<String, usize> {
        let mut topic_counts = HashMap::new();
        for reference in references {
            *topic_counts.entry(reference).or_insert(0) += 1;
        }
        topic_counts
    }

    /// Initialize local storage with files from GitHub
    pub async fn initialize_local_storage(
        github_service: &dyn GitHubService,
        settings: Arc<RwLock<Settings>>,
    ) -> Result<(), Box<dyn StdError + Send + Sync>> {
        // Validate GitHub settings
        let settings_guard = settings.read().await;
        let github_settings = &settings_guard.github;
        
        if github_settings.token.is_empty() {
            return Err("GitHub token is required".into());
        }
        if github_settings.owner.is_empty() {
            return Err("GitHub owner is required".into());
        }
        if github_settings.repo.is_empty() {
            return Err("GitHub repository name is required".into());
        }
        
        debug!("Using GitHub settings - Owner: {}, Repo: {}, Base Path: {}",
            github_settings.owner,
            github_settings.repo,
            github_settings.base_path
        );
        drop(settings_guard);

        // Check if we already have a valid local setup
        if Self::has_valid_local_setup() {
            info!("Valid local setup found, skipping initialization");
            return Ok(());
        }

        info!("Initializing local storage with files from GitHub");

        // Ensure directories exist and have proper permissions
        Self::ensure_directories()?;

        // Step 1: Get all markdown files from GitHub with proper error handling
        let github_files = match github_service.fetch_file_metadata(false).await {
            Ok(files) => {
                info!("Found {} markdown files in GitHub", files.len());
                files
            },
            Err(e) => {
                error!("Failed to fetch file metadata: {}", e);
                return Err(e);
            }
        };

        let mut file_sizes = HashMap::new();
        let mut file_contents = HashMap::new();
        let mut file_metadata = HashMap::new();
        let mut metadata_store = MetadataStore::new();
        let mut failed_files = Vec::new();

        // Process files in batches to prevent timeouts
        const BATCH_SIZE: usize = 5;
        for chunk in github_files.chunks(BATCH_SIZE) {
            let mut futures = Vec::new();
            
            for file_meta in chunk {
                let file_meta = file_meta.clone();
                let github_service = github_service;
                
                futures.push(async move {
                    match github_service.fetch_file_content(&file_meta.download_url).await {
                        Ok(content) => {
                            // Check if file is public
                            let first_line = content.lines().next().unwrap_or("").trim();
                            if first_line != "public:: true" {
                                debug!("Skipping non-public file: {}", file_meta.name);
                                return Ok(None);
                            }

                            let file_path = format!("{}/{}", MARKDOWN_DIR, file_meta.name);
                            if let Err(e) = fs::write(&file_path, &content) {
                                error!("Failed to write file {}: {}", file_path, e);
                                return Err(e.into());
                            }

                            Ok(Some((file_meta, content)))
                        }
                        Err(e) => {
                            error!("Failed to fetch content for {}: {}", file_meta.name, e);
                            Err(e)
                        }
                    }
                });
            }

            // Wait for batch to complete with timeout
            let results = futures::future::join_all(futures).await;
            
            for result in results {
                match result {
                    Ok(Some((file_meta, content))) => {
                        let node_name = file_meta.name.trim_end_matches(".md").to_string();
                        file_sizes.insert(node_name.clone(), content.len());
                        file_contents.insert(node_name, content.clone());
                        file_metadata.insert(file_meta.name.clone(), file_meta);
                    }
                    Ok(None) => continue, // Skipped non-public file
                    Err(e) => {
                        error!("Failed to process file in batch: {}", e);
                        failed_files.push(e.to_string());
                    }
                }
            }

            sleep(GITHUB_API_DELAY).await;
        }

        // Step 3: Process files and create metadata
        for (node_name, content) in &file_contents {
            let file_name = format!("{}.md", node_name);
            let local_sha1 = Self::calculate_sha1(content);

            // Extract references and create metadata
            let valid_nodes: Vec<String> = file_contents.keys().cloned().collect();
            let references = Self::extract_references(content, &valid_nodes);
            let topic_counts = Self::convert_references_to_topic_counts(references);

            // Get GitHub metadata
            let github_meta = file_metadata.get(&file_name).unwrap();
            let last_modified = github_meta.last_modified.unwrap_or_else(|| Utc::now());

            // Calculate node size
            let file_size = *file_sizes.get(node_name).unwrap();
            let node_size = Self::calculate_node_size(file_size);

            // Create metadata entry
            let metadata = Metadata {
                file_name: file_name.clone(),
                file_size,
                node_size,
                hyperlink_count: Self::count_hyperlinks(content),
                sha1: local_sha1,
                last_modified,
                perplexity_link: String::new(),
                last_perplexity_process: None,
                topic_counts,
            };

            metadata_store.insert(file_name, metadata);
        }

        // Step 4: Save metadata
        info!("Saving metadata for {} public files", metadata_store.len());
        Self::save_metadata(&metadata_store)?;

        info!("Initialization complete. Processed {} public files", metadata_store.len());
        Ok(())
    }

    /// Check if we have a valid local setup
    fn has_valid_local_setup() -> bool {
        if let Ok(metadata_content) = fs::read_to_string(METADATA_PATH) {
            if metadata_content.trim().is_empty() {
                return false;
            }
            
            if let Ok(metadata) = serde_json::from_str::<MetadataStore>(&metadata_content) {
                return metadata.validate_files(MARKDOWN_DIR);
            }
        }
        false
    }

    /// Ensures all required directories exist with proper permissions
    #[allow(dead_code)]
    fn ensure_directories() -> Result<(), Error> {
        // Create markdown directory
        let markdown_dir = Path::new(MARKDOWN_DIR);
        if !markdown_dir.exists() {
            info!("Creating markdown directory at {:?}", markdown_dir);
            fs::create_dir_all(markdown_dir)
                .map_err(|e| Error::new(std::io::ErrorKind::Other, format!("Failed to create markdown directory: {}", e)))?;
            // Set permissions to allow writing
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(markdown_dir, fs::Permissions::from_mode(0o777))
                    .map_err(|e| Error::new(std::io::ErrorKind::Other, format!("Failed to set markdown directory permissions: {}", e)))?;
            }
        }

        // Create metadata directory if it doesn't exist
        let metadata_dir = Path::new(METADATA_PATH).parent().unwrap();
        if !metadata_dir.exists() {
            info!("Creating metadata directory at {:?}", metadata_dir);
            fs::create_dir_all(metadata_dir)
                .map_err(|e| Error::new(std::io::ErrorKind::Other, format!("Failed to create metadata directory: {}", e)))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(metadata_dir, fs::Permissions::from_mode(0o777))
                    .map_err(|e| Error::new(std::io::ErrorKind::Other, format!("Failed to set metadata directory permissions: {}", e)))?;
            }
        }

        // Verify permissions by attempting to create a test file
        let test_file = format!("{}/test_permissions", MARKDOWN_DIR);
        match fs::write(&test_file, "test") {
            Ok(_) => {
                info!("Successfully wrote test file to {}", test_file);
                fs::remove_file(&test_file)
                    .map_err(|e| Error::new(std::io::ErrorKind::Other, format!("Failed to remove test file: {}", e)))?;
                info!("Successfully removed test file");
                info!("Directory permissions verified");
                Ok(())
            },
            Err(e) => {
                error!("Failed to verify directory permissions: {}", e);
                if let Ok(current_dir) = std::env::current_dir() {
                    error!("Current directory: {:?}", current_dir);
                }
                if let Ok(dir_contents) = fs::read_dir(MARKDOWN_DIR) {
                    error!("Directory contents: {:?}", dir_contents);
                }
                Err(Error::new(std::io::ErrorKind::PermissionDenied, format!("Failed to verify directory permissions: {}", e)))
            }
        }
    }

    /// Handles incremental updates after initial setup
    pub async fn fetch_and_process_files(
        &self,
        github_service: &dyn GitHubService,
        settings: Arc<RwLock<Settings>>,
        metadata_store: &mut MetadataStore,
    ) -> Result<Vec<ProcessedFile>, Error> {
        // Validate GitHub settings
        let settings_guard = settings.read().await;
        let github_settings = &settings_guard.github;
        
        if github_settings.token.is_empty() {
            return Err(Error::new(std::io::ErrorKind::Other, "GitHub token is required"));
        }
        if github_settings.owner.is_empty() {
            return Err(Error::new(std::io::ErrorKind::Other, "GitHub owner is required"));
        }
        if github_settings.repo.is_empty() {
            return Err(Error::new(std::io::ErrorKind::Other, "GitHub repository name is required"));
        }
        
        let base_path = github_settings.base_path.clone();
        debug!("Using GitHub settings - Owner: {}, Repo: {}, Base Path: {}",
            github_settings.owner,
            github_settings.repo,
            base_path
        );
        drop(settings_guard);

        // Construct the full path for the GitHub API request
        let api_path = if base_path.is_empty() {
            String::from("")
        } else {
            base_path.trim_matches('/').to_string()
        };

        // Ensure directories exist and have proper permissions
        if let Err(e) = Self::ensure_directories() {
            return Err(Error::new(std::io::ErrorKind::Other, e.to_string()));
        }

        // Get files from GitHub
        let github_files_metadata = match github_service.fetch_files(&api_path).await {
            Ok(files) => {
                info!("Found {} files in GitHub path: {}", files.len(), api_path);
                files
            }
            Err(e) => {
                error!("Failed to list files from GitHub: {}", e);
                return Err(Error::new(std::io::ErrorKind::Other, e.to_string()));
            }
        };

        let mut processed_files = Vec::new();

        // Remove files that no longer exist in GitHub
        let github_filenames: std::collections::HashSet<_> = github_files_metadata.iter()
            .map(|f| f.name.clone())
            .collect();

        // Remove files from metadata store that don't exist in GitHub anymore
        let files_to_remove: Vec<_> = metadata_store.keys()
            .filter(|file_name| !github_filenames.contains(*file_name))
            .cloned()
            .collect();

        for file_name in files_to_remove {
            debug!("Removing metadata for deleted file: {}", file_name);
            metadata_store.remove(&file_name);
        }

        // Get list of valid node names (filenames without .md)
        let valid_nodes: Vec<String> = github_files_metadata.iter()
            .map(|f| f.name.trim_end_matches(".md").to_string())
            .collect();

        // Process files that need updating
        let files_to_process: Vec<_> = github_files_metadata.into_iter()
            .filter(|file_meta| {
                let local_meta = metadata_store.get(&file_meta.name);
                local_meta.map_or(true, |meta| meta.sha1 != file_meta.sha)
            })
            .collect();

        // Process files in batches to prevent timeouts
        const BATCH_SIZE: usize = 5;
        for chunk in files_to_process.chunks(BATCH_SIZE) {
            let mut futures = Vec::new();
            
            for file_meta in chunk {
                let file_meta = file_meta.clone();
                let github_service = github_service;
                let valid_nodes = valid_nodes.clone();
                
                futures.push(async move {
                    match github_service.fetch_file_content(&file_meta.download_url).await {
                        Ok(content) => {
                            let first_line = content.lines().next().unwrap_or("").trim();
                            if first_line != "public:: true" {
                                debug!("Skipping non-public file: {}", file_meta.name);
                                return Ok(None);
                            }

                            let file_path = format!("{}/{}", MARKDOWN_DIR, file_meta.name);
                            if let Err(e) = fs::write(&file_path, &content) {
                                error!("Failed to write file {}: {}", file_path, e);
                                return Err(Error::new(std::io::ErrorKind::Other, e.to_string()));
                            }

                            // Extract references
                            let references = Self::extract_references(&content, &valid_nodes);
                            let topic_counts = Self::convert_references_to_topic_counts(references);

                            // Calculate node size
                            let file_size = content.len();
                            let node_size = Self::calculate_node_size(file_size);

                            let new_metadata = Metadata {
                                file_name: file_meta.name.clone(),
                                file_size,
                                node_size,
                                hyperlink_count: Self::count_hyperlinks(&content),
                                sha1: Self::calculate_sha1(&content),
                                last_modified: file_meta.last_modified.expect("Last modified time should be present"),
                                perplexity_link: String::new(),
                                last_perplexity_process: None,
                                topic_counts,
                            };

                            Ok(Some((file_meta.name, content, new_metadata)))
                        }
                        Err(e) => {
                            error!("Failed to fetch content for {}: {}", file_meta.name, e);
                            Err(Error::new(std::io::ErrorKind::Other, e.to_string()))
                        }
                    }
                });
            }

            // Wait for batch to complete
            let results = futures::future::join_all(futures).await;
            
            for result in results {
                match result {
                    Ok(Some((file_name, content, new_metadata))) => {
                        metadata_store.insert(file_name.clone(), new_metadata.clone());
                        processed_files.push(ProcessedFile {
                            file_name,
                            content,
                            is_public: true,
                            metadata: new_metadata,
                        });
                    }
                    Ok(None) => continue, // Skipped non-public file
                    Err(e) => {
                        error!("Failed to process file in batch: {}", e);
                    }
                }
            }

            sleep(GITHUB_API_DELAY).await;
        }

        // Save updated metadata
        if let Err(e) = Self::save_metadata(metadata_store) {
            return Err(Error::new(std::io::ErrorKind::Other, e.to_string()));
        }

        Ok(processed_files)
    }

    /// Save metadata to file
    pub fn save_metadata(metadata: &MetadataStore) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        fs::write(METADATA_PATH, json)
            .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    }

    /// Calculate SHA1 hash of content
    fn calculate_sha1(content: &str) -> String {
        use sha1::{Sha1, Digest};
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Count hyperlinks in content
    fn count_hyperlinks(content: &str) -> usize {
        let re = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
        re.find_iter(content).count()
    }
}
