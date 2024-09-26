// src/services/file_service.rs

use crate::models::metadata::Metadata;
use crate::config::{Settings, GitHubConfig};
use crate::services::perplexity_service::{PerplexityService, PerplexityServiceImpl, ApiClientImpl};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use async_trait::async_trait;
use log::{info, debug};
use regex::Regex;
use sha1::{Sha1, Digest};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Represents a file fetched from GitHub.
#[derive(Serialize, Deserialize, Clone)]
pub struct GithubFile {
    /// Name of the file (e.g., "example.md").
    pub name: String,
    /// Content of the file in Markdown format.
    pub content: String,
    /// SHA hash of the file content from GitHub.
    pub sha: String,
}

/// Represents a processed file after applying transformations.
#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessedFile {
    /// Name of the processed file.
    pub file_name: String,
    /// Processed content of the file.
    pub content: String,
}

/// Trait defining the GitHub service behavior.
#[async_trait]
pub trait GitHubService: Send + Sync {
    /// Fetches Markdown files from the specified GitHub repository.
    async fn fetch_files(&self) -> Result<Vec<GithubFile>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Represents a GitHub service that uses actual GitHub API calls.
pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
}

impl RealGitHubService {
    /// Creates a new instance of `RealGitHubService` using the provided GitHubConfig.
    pub fn new(config: GitHubConfig) -> Self {
        Self {
            client: Client::new(),
            token: config.github_access_token,
            owner: config.github_owner,
            repo: config.github_repo,
            base_path: config.github_directory,
        }
    }

    /// Fetches the contents of a specific directory from the GitHub repository.
    ///
    /// # Arguments
    ///
    /// * `path` - The directory path within the repository to fetch.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of JSON values representing the directory contents or an error.
    async fn fetch_directory_contents(&self, path: &str) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("https://api.github.com/repos/{}/{}/contents/{}", self.owner, self.repo, path);
        debug!("Fetching contents from GitHub: {}", url);

        let response = self.client.get(&url)
            .header("Authorization", format!("token {}", self.token))
            .header("User-Agent", "rust-github-api")
            .send()
            .await?;

        debug!("GitHub API response status: {}", response.status());

        let response_body = response.text().await?.to_string(); // Convert to String
        debug!("GitHub API response body: {}", response_body);

        let contents: Vec<serde_json::Value> = serde_json::from_str(&response_body)?;
        Ok(contents)
    }

    /// Fetches the content of a specific file from GitHub using its download URL.
    ///
    /// # Arguments
    ///
    /// * `download_url` - The download URL of the file to fetch.
    ///
    /// # Returns
    ///
    /// A `Result` containing the file content as a string or an error.
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
}

#[async_trait]
impl GitHubService for RealGitHubService {
    /// Fetches all Markdown files from the GitHub repository recursively.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `GithubFile` or an error.
    async fn fetch_files(&self) -> Result<Vec<GithubFile>, Box<dyn std::error::Error + Send + Sync>> {
        let mut github_files = Vec::new();
        let mut directories_to_process = vec![self.base_path.clone()];

        // Recursively fetch files from all directories
        while let Some(current_path) = directories_to_process.pop() {
            let contents = self.fetch_directory_contents(&current_path).await?;

            for item in contents {
                let name = item["name"].as_str().unwrap_or("");
                let item_type = item["type"].as_str().unwrap_or("");
                let path = item["path"].as_str().unwrap_or("");

                if item_type == "dir" {
                    // If the item is a directory, add it to the list to be processed
                    directories_to_process.push(path.to_string());
                } else if item_type == "file" && name.ends_with(".md") {
                    // If the item is a Markdown file, fetch its content
                    if let Some(download_url) = item["download_url"].as_str() {
                        debug!("Fetching content for file: {}", name);
                        let content = self.fetch_file_content(download_url).await?;
                        let sha = item["sha"].as_str().unwrap_or("").to_string();

                        // Add the file to the list of GitHub files
                        github_files.push(GithubFile {
                            name: name.to_string(),
                            content,
                            sha,
                        });
                        debug!("Added file to github_files: {}", name);
                    }
                } else {
                    debug!("Skipping non-markdown file: {}", name);
                }
            }
        }

        debug!("Fetched {} markdown files from GitHub", github_files.len());
        Ok(github_files)
    }
}

/// Service responsible for handling file operations, including fetching from GitHub and processing.
pub struct FileService;

impl FileService {
    /// Fetches Markdown files from GitHub and processes them.
    ///
    /// # Arguments
    ///
    /// * `github_service` - An instance of a service that implements the `GitHubService` trait.
    /// * `settings` - Application settings containing configuration data.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `ProcessedFile` on success or an error on failure.
    pub async fn fetch_and_process_files(
        github_service: &dyn GitHubService,
        settings: &Settings,
    ) -> Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>> {
        // Step 1: Fetch files from GitHub
        let github_files = github_service.fetch_files().await?;
        debug!("Fetched {} files from GitHub", github_files.len());

        // Step 2: Process the fetched files
        let processed_files = Self::process_files(github_files, settings).await?;
        debug!("Processed {} files", processed_files.len());

        Ok(processed_files)
    }

    /// Processes the fetched GitHub files. This includes:
    /// 1. Checking if a file needs processing.
    /// 2. Stripping Logseq-style double brackets.
    /// 3. Associating content with topics.
    /// 4. Enhancing content using the PerplexityService.
    /// 5. Updating local metadata.
    async fn process_files(
        github_files: Vec<GithubFile>,
        settings: &Settings,
    ) -> Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>> {
        let mut processed_files = Vec::new();
        let local_metadata = Self::load_local_metadata()?;
        debug!("Loaded {} metadata entries", local_metadata.len());

        // Create an instance of your PerplexityService implementation
        let perplexity_service = PerplexityServiceImpl;

        // Create an instance of your ApiClient
        let api_client = ApiClientImpl::new();

        for file in github_files {
            if Self::should_process_file(&file, &local_metadata) {
                debug!("Processing file: {}", file.name);

                let stripped_content = Self::strip_double_brackets(&file.content);
                let processed_content = Self::process_against_topics(&stripped_content, &settings.topics);

                // Use the PerplexityService to process the content
                let processed_file = perplexity_service.process_file(processed_content, settings, &api_client).await?; 
                
                // Update local metadata 
                let new_metadata = Metadata {
                    file_name: file.name.clone(),
                    last_modified: chrono::Utc::now(),
                    processed_file: processed_file.content.clone(), // Use processed content
                    original_file: file.content,
                };
                Self::save_file_metadata(new_metadata)?;

                processed_files.push(processed_file);
            } else {
                debug!("Skipping file: {}", file.name);
            }
        }

        Ok(processed_files)
    }

    /// Determines whether a file should be processed based on its metadata.
    ///
    /// # Arguments
    ///
    /// * `file` - A reference to the `GithubFile` to evaluate.
    /// * `local_metadata` - A reference to the local metadata hashmap.
    ///
    /// # Returns
    ///
    /// `true` if the file should be processed; otherwise, `false`.
    fn should_process_file(file: &GithubFile, local_metadata: &HashMap<String, Metadata>) -> bool {
        // Check if the first line indicates it's a public file
        let first_line = file.content.lines().next().unwrap_or("").trim();
        if first_line != "public:: true" {
            return false;
        }

        // Calculate SHA1 hashes to determine if the file has been modified
        let local_sha = local_metadata.get(&file.name).map(|m| Self::calculate_sha1(&m.original_file));
        let github_sha = Self::calculate_sha1(&file.content);

        // Process the file if it's new or has been modified
        local_sha.map_or(true, |local| local != github_sha)
    }

    /// Calculates the SHA1 hash of the given content.
    ///
    /// # Arguments
    ///
    /// * `content` - The string content to hash.
    ///
    /// # Returns
    ///
    /// A hexadecimal string representation of the SHA1 hash.
    fn calculate_sha1(content: &str) -> String {
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Loads local metadata from a JSON file.
    ///
    /// # Returns
    ///
    /// A `Result` containing a hashmap of metadata or an error.
    fn load_local_metadata() -> Result<HashMap<String, Metadata>, Box<dyn std::error::Error + Send + Sync>> {
        // Define the path to the metadata file
        let metadata_path = "/app/data/markdown/metadata.json";

        // Check if the metadata file exists
        if Path::new(metadata_path).exists() {
            // Read the metadata file
            let metadata_content = fs::read_to_string(metadata_path)?;
            // Deserialize the JSON content into a hashmap
            let metadata: HashMap<String, Metadata> = serde_json::from_str(&metadata_content)?;
            Ok(metadata)
        } else {
            // If the metadata file doesn't exist, return an empty hashmap
            Ok(HashMap::new())
        }
    }

    /// Strips double brackets [[ ]] from the content using regular expressions.
    ///
    /// # Arguments
    ///
    /// * `content` - The string content from which to remove double brackets.
    ///
    /// # Returns
    ///
    /// A new string with double brackets removed.
    fn strip_double_brackets(content: &str) -> String {
        let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
        re.replace_all(content, "$1").to_string()
    }

    /// Associates the processed content with relevant topics based on the topics list.
    ///
    /// # Arguments
    ///
    /// * `content` - The stripped content of the file.
    /// * `topics` - A slice of topic strings to associate with the content.
    ///
    /// # Returns
    ///
    /// A new string with topic associations appended.
    fn process_against_topics(content: &str, topics: &[String]) -> String {
        let mut processed_content = content.to_string();
        for topic in topics {
            if content.contains(topic) {
                processed_content.push_str(&format!("\nRelated to topic: {}", topic));
            }
        }
        processed_content
    }

    /// Saves the processed Markdown file to the persistent volume.
    ///
    /// # Arguments
    ///
    /// * `metadata` - The `Metadata` instance containing file information.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    pub fn save_file_metadata(metadata: Metadata) -> Result<(), std::io::Error> {
        info!("Saving metadata for file: {}", metadata.file_name);

        // Define the path where the processed Markdown file will be saved
        let markdown_path = format!("/app/data/markdown/{}", metadata.file_name);

        // Ensure the markdown directory exists; if not, create it
        if let Some(parent) = Path::new(&markdown_path).parent() {
            fs::create_dir_all(parent)?;
            debug!("Ensured directory exists: {}", parent.display());
        }

        // Write the processed content to the Markdown file
        fs::write(&markdown_path, &metadata.processed_file)?;
        debug!("Written processed content to: {}", markdown_path);

        // Update the metadata JSON file
        Self::update_metadata_file(&metadata)?;

        Ok(())
    }

    /// Updates the metadata JSON file with the latest file metadata.
    ///
    /// # Arguments
    ///
    /// * `metadata` - The `Metadata` instance to be saved.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    fn update_metadata_file(metadata: &Metadata) -> Result<(), std::io::Error> {
        // Define the path to the metadata file
        let metadata_path = "/app/data/markdown/metadata.json";

        // Load existing metadata
        let mut metadata_map = if Path::new(metadata_path).exists() {
            let content = fs::read_to_string(metadata_path)?;
            serde_json::from_str::<HashMap<String, Metadata>>(&content)?
        } else {
            HashMap::new()
        };

        // Update the metadata map with the new metadata
        metadata_map.insert(metadata.file_name.clone(), metadata.clone());

        // Serialize the updated metadata map to JSON
        let updated_content = serde_json::to_string_pretty(&metadata_map)?;

        // Write the updated metadata back to the metadata file
        fs::write(metadata_path, updated_content)?;
        debug!("Updated metadata file at: {}", metadata_path);

        Ok(())
    }
}