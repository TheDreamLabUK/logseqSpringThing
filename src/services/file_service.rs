use crate::models::metadata::Metadata;
use crate::config::Settings;
use serde::{Deserialize, Serialize};
use reqwest::{Client, header::{HeaderMap, HeaderValue, IF_NONE_MATCH, ETAG}};
use async_trait::async_trait;
use log::{info, debug, error, warn};
use regex::Regex;
use sha1::{Sha1, Digest};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use chrono::{Utc, DateTime};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::error::Error as StdError;
use futures::stream::{self, StreamExt};
use std::time::Duration;
use tokio::time::sleep;

const METADATA_PATH: &str = "/app/data/markdown/metadata.json";
const MARKDOWN_DIR: &str = "/app/data/markdown";
const CACHE_DURATION: Duration = Duration::from_secs(300); // 5 minutes
const MAX_CONCURRENT_DOWNLOADS: usize = 5;
const GITHUB_API_DELAY: Duration = Duration::from_millis(100); // Rate limiting delay

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
    pub download_url: String,
    #[serde(skip)]
    pub etag: Option<String>,
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub last_checked: Option<DateTime<Utc>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedFile {
    pub file_name: String,
    pub content: String,
    pub is_public: bool,
    pub metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize)]
struct TreeResponse {
    sha: String,
    tree: Vec<TreeItem>,
    truncated: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct TreeItem {
    path: String,
    mode: String,
    #[serde(rename = "type")]
    item_type: String,
    sha: String,
    url: Option<String>,
}

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>>;
    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>>;
    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>>;
    async fn check_file_public_status(&self, download_url: &str) -> Result<bool, Box<dyn StdError + Send + Sync>>;
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
    metadata_cache: Arc<RwLock<HashMap<String, GithubFileMetadata>>>,
}

impl RealGitHubService {
    pub async fn new(settings: Arc<RwLock<Settings>>) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let settings = settings.read().await;
        let github_settings = &settings.github;
        if github_settings.github_access_token.is_empty() {
            return Err("GitHub access token is empty".into());
        }

        let mut headers = HeaderMap::new();
        headers.insert("User-Agent", HeaderValue::from_static("rust-github-api"));
        
        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            token: github_settings.github_access_token.clone(),
            owner: github_settings.github_owner.clone(),
            repo: github_settings.github_repo.clone(),
            base_path: github_settings.github_directory.clone(),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Fetches repository tree using the Git Trees API for better performance
    async fn fetch_tree(&self) -> Result<TreeResponse, Box<dyn StdError + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/git/trees/HEAD",
            self.owner, self.repo
        );

        let response = self.client.get(&url)
            .header("Authorization", format!("token {}", self.token))
            .query(&[("recursive", "1")])
            .send()
            .await?;

        let tree: TreeResponse = response.json().await?;
        Ok(tree)
    }

    /// Fetches only markdown files from the root directory using the Trees API
    async fn fetch_directory_contents(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        // First, check the cache
        {
            let cache = self.metadata_cache.read().await;
            let now = Utc::now();
            
            // If cache is fresh and not empty, use it
            if !cache.is_empty() {
                if let Some(first_item) = cache.values().next() {
                    if let Some(last_checked) = first_item.last_checked {
                        if (now - last_checked) < chrono::Duration::from_std(CACHE_DURATION).unwrap() {
                            debug!("Using cached metadata for files");
                            return Ok(cache.values().cloned().collect());
                        }
                    }
                }
            }
        }

        // Cache is stale or empty, fetch from GitHub
        let tree = self.fetch_tree().await?;
        
        let markdown_files: Vec<GithubFileMetadata> = tree.tree.into_iter()
            .filter(|item| {
                let is_file = item.item_type == "blob";
                let path = Path::new(&item.path);
                let is_in_root = path.parent().map_or(true, |p| p == Path::new(""));
                let is_markdown = item.path.ends_with(".md");
                is_file && is_in_root && is_markdown
            })
            .map(|item| {
                let download_url = format!(
                    "https://raw.githubusercontent.com/{}/{}/HEAD/{}",
                    self.owner, self.repo, item.path
                );
                GithubFileMetadata {
                    name: item.path,
                    sha: item.sha,
                    download_url,
                    etag: None,
                    last_checked: Some(Utc::now()),
                }
            })
            .collect();

        // Update cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.clear();
            for metadata in &markdown_files {
                cache.insert(metadata.name.clone(), metadata.clone());
            }
        }

        debug!("Found {} markdown files in root directory", markdown_files.len());
        Ok(markdown_files)
    }

    /// Efficiently checks if a file is public by only downloading the first chunk
    async fn check_file_public_status(&self, download_url: &str) -> Result<bool, Box<dyn StdError + Send + Sync>> {
        // Don't use Range header as it might interfere with line endings
        let response = self.client.get(download_url)
            .header("Authorization", format!("token {}", self.token))
            .send()
            .await?;

        let content = response.text().await?;
        let bytes = content.as_bytes();
        
        // Get raw bytes up to first newline
        let first_line = bytes
            .iter()
            .take_while(|&&b| b != b'\n' && b != b'\r')
            .copied()
            .collect::<Vec<u8>>();

        // Convert to string and check
        let first_line = String::from_utf8_lossy(&first_line);
        let is_public = first_line.trim() == "public:: true";
        
        debug!("Raw first line: {:?}, is_public: {}", first_line, is_public);
        Ok(is_public)
    }

    async fn fetch_file_content(&self, download_url: &str) -> Result<String, Box<dyn StdError + Send + Sync>> {
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_str(&format!("token {}", self.token))?);

        // Get cached ETag if available
        let etag = {
            let cache = self.metadata_cache.read().await;
            cache.values()
                .find(|m| m.download_url == download_url)
                .and_then(|m| m.etag.clone())
        };

        if let Some(etag) = etag {
            headers.insert(IF_NONE_MATCH, HeaderValue::from_str(&etag)?);
        }

        let response = self.client.get(download_url)
            .headers(headers)
            .send()
            .await?;

        // Update ETag in cache if provided
        if let Some(new_etag) = response.headers().get(ETAG) {
            let mut cache = self.metadata_cache.write().await;
            if let Some(metadata) = cache.values_mut().find(|m| m.download_url == download_url) {
                metadata.etag = Some(new_etag.to_str()?.to_string());
            }
        }

        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            // Use cached content
            let path = format!("{}/{}", MARKDOWN_DIR, download_url.split('/').last().unwrap_or(""));
            if let Ok(content) = fs::read_to_string(&path) {
                return Ok(content);
            }
        }

        let content = response.text().await?;
        Ok(content)
    }

    pub async fn fetch_file_metadata(&self) -> Result<Vec<GithubFileMetadata>, Box<dyn StdError + Send + Sync>> {
        self.fetch_directory_contents().await
    }

    async fn get_download_url(&self, file_name: &str) -> Result<Option<String>, Box<dyn StdError + Send + Sync>> {
        // Check cache first
        {
            let cache = self.metadata_cache.read().await;
            if let Some(metadata) = cache.get(file_name) {
                return Ok(Some(metadata.download_url.clone()));
            }
        }

        let url = format!("https://api.github.com/repos/{}/{}/contents/{}/{}", 
            self.owner, self.repo, self.base_path, file_name);

        let response = self.client.get(&url)
            .header("Authorization", format!("token {}", self.token))
            .send()
            .await?;

        if response.status().is_success() {
            let json: serde_json::Value = response.json().await?;
            Ok(json["download_url"].as_str().map(|s| s.to_string()))
        } else {
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

    async fn check_file_public_status(&self, download_url: &str) -> Result<bool, Box<dyn StdError + Send + Sync>> {
        self.check_file_public_status(download_url).await
    }
}

pub struct FileService;

impl FileService {
    /// Ensures all required directories exist
    fn ensure_directories() -> Result<(), std::io::Error> {
        debug!("Ensuring required directories exist");
        fs::create_dir_all(MARKDOWN_DIR)?;
        if let Some(parent_dir) = Path::new(METADATA_PATH).parent() {
            fs::create_dir_all(parent_dir)?;
        }
        Ok(())
    }

    /// Optimized file processing:
    /// 1. Uses Trees API for efficient file listing
    /// 2. Implements caching with ETags
    /// 3. Parallel downloads with rate limiting
    /// 4. Conditional requests to minimize bandwidth
    pub async fn fetch_and_process_files(
        github_service: &dyn GitHubService,
        _settings: Arc<RwLock<Settings>>,
        metadata_map: &mut HashMap<String, Metadata>,
    ) -> Result<Vec<ProcessedFile>, Box<dyn StdError + Send + Sync>> {
        // Ensure directories exist before any operations
        Self::ensure_directories()?;

        // Get metadata for markdown files in root directory
        let github_files_metadata = github_service.fetch_file_metadata().await?;
        debug!("Fetched metadata for {} markdown files", github_files_metadata.len());

        let mut processed_files = Vec::new();
        let local_metadata = metadata_map.clone();
        
        // Clean up removed files
        let github_file_names: HashSet<String> = github_files_metadata.iter()
            .map(|f| f.name.clone())
            .collect();
        
        let removed_files: Vec<String> = local_metadata.keys()
            .filter(|name| !github_file_names.contains(*name))
            .cloned()
            .collect();
        
        for removed_file in removed_files {
            info!("Removing file not present on GitHub: {}", removed_file);
            metadata_map.remove(&removed_file);
            // Also remove the local file if it exists
            let file_path = format!("{}/{}", MARKDOWN_DIR, removed_file);
            if Path::new(&file_path).exists() {
                if let Err(e) = fs::remove_file(&file_path) {
                    error!("Failed to remove file {}: {}", file_path, e);
                }
            }
        }

        // Process files in parallel with rate limiting
        let files_to_process: Vec<_> = github_files_metadata.into_iter()
            .filter(|file_meta| {
                let local_meta = metadata_map.get(&file_meta.name);
                local_meta.map_or(true, |local_meta| local_meta.sha1 != file_meta.sha)
            })
            .collect();

        let results = stream::iter(files_to_process)
            .map(|file_meta| {
                let github_service = github_service;
                async move {
                    // Add delay for rate limiting
                    sleep(GITHUB_API_DELAY).await;

                    // Check if file is public
                    match github_service.check_file_public_status(&file_meta.download_url).await {
                        Ok(is_public) => {
                            if !is_public {
                                warn!("Skipping non-public file: {} (first line did not match 'public:: true')", file_meta.name);
                                return Ok(None);
                            }
                            
                            // Download content for public files
                            match github_service.fetch_file_content(&file_meta.download_url).await {
                                Ok(content) => {
                                    let file_path = format!("{}/{}", MARKDOWN_DIR, file_meta.name);
                                    fs::write(&file_path, &content)?;
                                    
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
                                    
                                    Ok(Some(ProcessedFile {
                                        file_name: file_meta.name,
                                        content,
                                        is_public: true,
                                        metadata: new_metadata,
                                    }))
                                }
                                Err(e) => {
                                    error!("Failed to fetch content: {}", e);
                                    Err(e)
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to check public status: {}", e);
                            Err(e)
                        }
                    }
                }
            })
            .buffer_unordered(MAX_CONCURRENT_DOWNLOADS)
            .collect::<Vec<_>>()
            .await;

        // Process results
        for result in results {
            match result {
                Ok(Some(processed_file)) => {
                    let file_name = processed_file.file_name.clone(); // Clone before moving
                    metadata_map.insert(file_name.clone(), processed_file.metadata.clone());
                    processed_files.push(processed_file);
                    debug!("Successfully processed public file: {}", file_name);
                }
                Ok(None) => {} // Skip non-public files
                Err(e) => error!("Error processing file: {}", e),
            }
        }

        debug!("Processed {} files after optimization", processed_files.len());
        Ok(processed_files)
    }

    pub fn load_or_create_metadata() -> Result<HashMap<String, Metadata>, Box<dyn StdError + Send + Sync>> {
        // Ensure directories exist before attempting to read/write metadata
        Self::ensure_directories()?;

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
        // Ensure directories exist before saving
        Self::ensure_directories()?;
        
        let updated_content = serde_json::to_string_pretty(metadata_map)?;
        fs::write(METADATA_PATH, updated_content)?;
        debug!("Updated metadata file at: {}", METADATA_PATH);
        Ok(())
    }

    pub fn update_metadata(metadata_map: &HashMap<String, Metadata>) -> Result<(), Box<dyn StdError + Send + Sync>> {
        // Ensure directories exist before updating
        Self::ensure_directories()?;

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
        // Ensure directories exist before saving
        Self::ensure_directories()?;

        info!("Saving metadata for file: {}", metadata.file_name);
        let markdown_path = format!("{}/{}", MARKDOWN_DIR, metadata.file_name);
        fs::write(&markdown_path, &metadata.perplexity_link)?;
        debug!("Written processed content to: {}", markdown_path);

        Self::update_metadata_file(&metadata)?;
        Ok(())
    }

    fn update_metadata_file(metadata: &Metadata) -> Result<(), std::io::Error> {
        // Ensure directories exist before updating
        Self::ensure_directories()?;

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
