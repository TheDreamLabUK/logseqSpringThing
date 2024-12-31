use crate::models::metadata::{Metadata, MetadataStore, MetadataOps};
use crate::models::graph::GraphData;
use crate::config::Settings;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use async_trait::async_trait;
use log::{info, debug, error};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use chrono::{Utc, DateTime};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::error::Error as StdError;
use std::time::Duration;
use tokio::time::sleep;
use actix_web::web;

// Constants
const METADATA_PATH: &str = "/app/data/markdown/metadata.json";
pub const MARKDOWN_DIR: &str = "/app/data/markdown";
const GITHUB_API_DELAY: Duration = Duration::from_millis(100); // Rate limiting delay
const MIN_NODE_SIZE: f64 = 5.0;
const MAX_NODE_SIZE: f64 = 50.0;

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

// Structure to hold reference information
#[derive(Default)]
struct ReferenceInfo {
    direct_mentions: usize,
}

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn fetch_file_metadata(&self, skip_debug_filter: bool) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>>;
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>>;
    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>>;
    async fn get_file_last_modified(&self, file_path: &str) -> Result<DateTime<Utc>, Box<dyn StdError + Send + Sync>>;
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
    _settings: Arc<RwLock<Settings>>,
}

impl RealGitHubService {
    pub fn new(
        token: String,
        owner: String,
        repo: String,
        base_path: String,
        _settings: Arc<RwLock<Settings>>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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
            _settings,
        })
    }
}

#[async_trait]
impl GitHubService for RealGitHubService {
    async fn fetch_file_metadata(&self, skip_debug_filter: bool) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        let url = if self.base_path.is_empty() {
            format!(
                "https://api.github.com/repos/{}/{}/contents",
                self.owner, self.repo
            )
        } else {
            format!(
                "https://api.github.com/repos/{}/{}/contents/{}",
                self.owner, self.repo, self.base_path
            )
        };
        
        debug!("Fetching GitHub metadata from URL: {}", url);

        // Set headers exactly as in the working curl command
        let response = self.client.get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        // Get status and headers for debugging
        let status = response.status();
        let headers = response.headers().clone();
        
        debug!("GitHub API response status: {}", status);
        debug!("GitHub API response headers: {:?}", headers);

        // Get response body
        let body = response.text().await?;
        
        // Log the first 1000 characters of the response for debugging
        debug!("GitHub API response preview: {}", &body[..body.len().min(1000)]);

        // Check for error response
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

        // Parse response as array
        let contents: Vec<serde_json::Value> = match serde_json::from_str(&body) {
            Ok(parsed) => parsed,
            Err(e) => {
                error!("Failed to parse GitHub API response: {}", e);
                error!("Response body: {}", body);
                return Err(Box::new(e));
            }
        };

        let settings = self._settings.read().await;
        let debug_enabled = settings.system.debug.enabled;
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
                
                let last_modified = match self.get_file_last_modified(&format!("{}/{}", self.base_path, name)).await {
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

        debug!("Found {} markdown files", markdown_files.len());
        Ok(markdown_files)
    }

    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>> {
        let url = if self.base_path.is_empty() {
            format!("https://api.github.com/repos/{}/{}/contents/{}", 
                self.owner, self.repo, file_name)
        } else {
            format!("https://api.github.com/repos/{}/{}/contents/{}/{}", 
                self.owner, self.repo, self.base_path, file_name)
        };

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
        let response = self.client.get(download_url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await?;
            error!("Failed to fetch file content. Status: {}, Error: {}", status, error_text);
            return Err(format!("Failed to fetch file content: {}", error_text).into());
        }

        let content = response.text().await?;
        Ok(content)
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
    settings: Arc<RwLock<Settings>>,
}

impl FileService {
    pub fn new(settings: Arc<RwLock<Settings>>) -> Self {
        Self { settings }
    }

    /// Process uploaded file and return graph data
    pub async fn process_file_upload(&self, payload: web::Bytes) -> Result<GraphData, Box<dyn StdError + Send + Sync>> {
        let content = String::from_utf8(payload.to_vec())?;
        let metadata = Self::load_or_create_metadata()?;
        let mut graph_data = GraphData::new();
        
        // Create a temporary file to process
        let temp_filename = format!("temp_{}.md", Utc::now().timestamp());
        let temp_path = format!("{}/{}", MARKDOWN_DIR, temp_filename);
        fs::write(&temp_path, &content)?;

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
    pub async fn list_files(&self) -> Result<Vec<String>, Box<dyn StdError + Send + Sync>> {
        let metadata = Self::load_or_create_metadata()?;
        Ok(metadata.keys().cloned().collect())
    }

    /// Load a specific file and return graph data
    pub async fn load_file(&self, filename: &str) -> Result<GraphData, Box<dyn StdError + Send + Sync>> {
        let file_path = format!("{}/{}", MARKDOWN_DIR, filename);
        if !Path::new(&file_path).exists() {
            return Err(format!("File not found: {}", filename).into());
        }

        let content = fs::read_to_string(&file_path)?;
        let metadata = Self::load_or_create_metadata()?;
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
    pub fn load_or_create_metadata() -> Result<MetadataStore, Box<dyn StdError + Send + Sync>> {
        if Path::new(METADATA_PATH).exists() {
            let content = fs::read_to_string(METADATA_PATH)?;
            if !content.trim().is_empty() {
                return Ok(serde_json::from_str(&content)?);
            }
        }
        Ok(MetadataStore::new())
    }

    /// Calculate node size based on file size
    fn calculate_node_size(file_size: usize) -> f64 {
        // Use logarithmic scaling for node size
        let size = if file_size == 0 {
            MIN_NODE_SIZE
        } else {
            let log_size = (file_size as f64).ln();
            let min_log = 0f64;
            let max_log = (100_000f64).ln(); // Assuming 100KB as max expected size
            
            let normalized = (log_size - min_log) / (max_log - min_log);
            MIN_NODE_SIZE + normalized * (MAX_NODE_SIZE - MIN_NODE_SIZE)
        };
        
        size.max(MIN_NODE_SIZE).min(MAX_NODE_SIZE)
    }

    /// Extract references to other files based on their names (case insensitive)
    fn extract_references(content: &str, valid_nodes: &[String]) -> HashMap<String, ReferenceInfo> {
        let mut references = HashMap::new();
        let content_lower = content.to_lowercase();
        
        for node_name in valid_nodes {
            let mut ref_info = ReferenceInfo::default();
            let node_name_lower = node_name.to_lowercase();
            
            // Create a regex pattern with word boundaries
            let pattern = format!(r"\b{}\b", regex::escape(&node_name_lower));
            if let Ok(re) = Regex::new(&pattern) {
                // Count case-insensitive matches of the filename
                let count = re.find_iter(&content_lower).count();
                
                // If we found any references, add them to the map
                if count > 0 {
                    debug!("Found {} references to {} in content", count, node_name);
                    ref_info.direct_mentions = count;
                    references.insert(format!("{}.md", node_name), ref_info);
                }
            }
        }
        
        references
    }

    fn convert_references_to_topic_counts(references: HashMap<String, ReferenceInfo>) -> HashMap<String, usize> {
        references.into_iter()
            .map(|(name, info)| {
                debug!("Converting reference for {} with {} mentions", name, info.direct_mentions);
                (name, info.direct_mentions)
            })
            .collect()
    }

    /// Initialize the local markdown directory and metadata structure.
    pub async fn initialize_local_storage(
        github_service: &dyn GitHubService,
        _settings: Arc<RwLock<Settings>>,
    ) -> Result<(), Box<dyn StdError + Send + Sync>> {
        info!("Checking local storage status");
        
        // Ensure required directories exist
        Self::ensure_directories()?;

        // Check if we already have a valid local setup
        if Self::has_valid_local_setup() {
            info!("Valid local setup found, skipping initialization");
            return Ok(());
        }

        info!("Initializing local storage with files from GitHub");

        // Step 1: Get all markdown files from GitHub
        let github_files = github_service.fetch_file_metadata(false).await?;
        info!("Found {} markdown files in GitHub", github_files.len());

        let mut file_sizes = HashMap::new();
        let mut file_contents = HashMap::new();
        let mut file_metadata = HashMap::new();
        let mut metadata_store = MetadataStore::new();
        
        // Step 2: First pass - collect all files and their contents
        for file_meta in github_files {
            match github_service.fetch_file_content(&file_meta.download_url).await {
                Ok(content) => {
                    // Check if file starts with "public:: true"
                    let first_line = content.lines().next().unwrap_or("").trim();
                    if first_line != "public:: true" {
                        debug!("Skipping non-public file: {}", file_meta.name);
                        continue;
                    }

                    let node_name = file_meta.name.trim_end_matches(".md").to_string();
                    file_sizes.insert(node_name.clone(), content.len());
                    file_contents.insert(node_name, content);
                    file_metadata.insert(file_meta.name.clone(), file_meta);
                }
                Err(e) => {
                    error!("Failed to fetch content for {}: {}", file_meta.name, e);
                }
            }
            sleep(GITHUB_API_DELAY).await;
        }

        // Get list of valid node names (filenames without .md)
        let valid_nodes: Vec<String> = file_contents.keys().cloned().collect();

        // Step 3: Second pass - extract references and create metadata
        for (node_name, content) in &file_contents {
            let file_name = format!("{}.md", node_name);
            let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
            
            // Calculate SHA1 of content
            let local_sha1 = Self::calculate_sha1(content);
            
            // Save file content
            fs::write(&file_path, content)?;

            // Extract references
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
    fn ensure_directories() -> Result<(), Box<dyn StdError + Send + Sync>> {
        // Create parent data directory first
        let data_dir = Path::new("/app/data");
        if !data_dir.exists() {
            info!("Creating data directory at {:?}", data_dir);
            fs::create_dir_all(data_dir)?;
            // Set permissions to allow writing
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(data_dir, fs::Permissions::from_mode(0o777))?;
            }
        }

        // Create markdown directory
        let markdown_dir = Path::new(MARKDOWN_DIR);
        if !markdown_dir.exists() {
            info!("Creating markdown directory at {:?}", markdown_dir);
            fs::create_dir_all(markdown_dir)?;
            // Set permissions to allow writing
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(markdown_dir, fs::Permissions::from_mode(0o777))?;
            }
        }

        // Create metadata directory if it doesn't exist
        let metadata_dir = Path::new(METADATA_PATH).parent().unwrap();
        if !metadata_dir.exists() {
            info!("Creating metadata directory at {:?}", metadata_dir);
            fs::create_dir_all(metadata_dir)?;
            // Set permissions to allow writing
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(metadata_dir, fs::Permissions::from_mode(0o777))?;
            }
        }

        // Verify permissions by attempting to create a test file
        let test_file = format!("{}/test_permissions", MARKDOWN_DIR);
        match fs::write(&test_file, "test") {
            Ok(_) => {
                info!("Successfully wrote test file to {}", test_file);
                fs::remove_file(&test_file)?;
                info!("Successfully removed test file");
                info!("Directory permissions verified");
                Ok(())
            },
            Err(e) => {
                error!("Failed to verify directory permissions: {}", e);
                error!("Current directory: {:?}", std::env::current_dir()?);
                error!("Directory contents: {:?}", fs::read_dir(MARKDOWN_DIR)?);
                Err(Box::new(e))
            }
        }
    }

    /// Handles incremental updates after initial setup
    pub async fn fetch_and_process_files(
        &self,
        github_service: &dyn GitHubService,
        _settings: Arc<RwLock<Settings>>,
        metadata_store: &mut MetadataStore,
    ) -> Result<Vec<ProcessedFile>, Box<dyn StdError + Send + Sync>> {
        // Ensure directories exist before any operations
        Self::ensure_directories()?;

        // Get metadata for markdown files in target directory
        let settings = self.settings.read().await;
        let skip_debug_filter = !settings.system.debug.enabled;
        drop(settings);
        let github_files_metadata = github_service.fetch_file_metadata(skip_debug_filter).await?;
        debug!("Fetched metadata for {} markdown files", github_files_metadata.len());

        let mut processed_files = Vec::new();

        // Save current metadata
        Self::save_metadata(metadata_store)?;

        // Clean up local files that no longer exist in GitHub
        let github_files: HashSet<_> = github_files_metadata.iter()
            .map(|meta| meta.name.clone())
            .collect();

        let local_files: HashSet<_> = metadata_store.keys().cloned().collect();
        let removed_files: Vec<_> = local_files.difference(&github_files).collect();

        for file_name in removed_files {
            let file_path = format!("{}/{}", MARKDOWN_DIR, file_name);
            if let Err(e) = fs::remove_file(&file_path) {
                error!("Failed to remove file {}: {}", file_path, e);
            }
            metadata_store.remove(file_name);
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

        // Process each file
        for file_meta in files_to_process {
            match github_service.fetch_file_content(&file_meta.download_url).await {
                Ok(content) => {
                    let first_line = content.lines().next().unwrap_or("").trim();
                    if first_line != "public:: true" {
                        debug!("Skipping non-public file: {}", file_meta.name);
                        continue;
                    }

                    let file_path = format!("{}/{}", MARKDOWN_DIR, file_meta.name);
                    fs::write(&file_path, &content)?;

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

                    metadata_store.insert(file_meta.name.clone(), new_metadata.clone());
                    processed_files.push(ProcessedFile {
                        file_name: file_meta.name,
                        content,
                        is_public: true,
                        metadata: new_metadata,
                    });
                }
                Err(e) => {
                    error!("Failed to fetch content: {}", e);
                }
            }
            sleep(GITHUB_API_DELAY).await;
        }

        // Save updated metadata
        Self::save_metadata(metadata_store)?;

        Ok(processed_files)
    }

    /// Save metadata to file
    pub fn save_metadata(metadata: &MetadataStore) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(METADATA_PATH, json)?;
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
