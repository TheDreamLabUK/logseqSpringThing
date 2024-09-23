use crate::models::metadata::Metadata;
use crate::config::Settings;
use serde::{Deserialize, Serialize};
use dotenv::dotenv;
use std::env;
use reqwest::Client;
use async_trait::async_trait;
use log::{info, debug};
use regex::Regex;
use sha1::{Sha1, Digest};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct GithubFile {
    pub name: String,
    pub content: String,
    pub sha: String,
}

#[derive(Serialize, Deserialize, Default)]
pub struct ProcessedFile {
    pub content: String,
}

#[async_trait]
pub trait GitHubService: Send + Sync {
    async fn fetch_files(&self) -> Result<Vec<GithubFile>, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct FileService;

impl FileService {
    pub async fn fetch_and_process_files(github_service: &dyn GitHubService, settings: &Settings) -> Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>> {
        let github_files = github_service.fetch_files().await?;
        debug!("Fetched {} files from GitHub", github_files.len());
        let processed_files = Self::process_files(github_files, settings).await?;
        debug!("Processed {} files", processed_files.len());
        Ok(processed_files)
    }

    async fn process_files(github_files: Vec<GithubFile>, settings: &Settings) -> Result<Vec<ProcessedFile>, Box<dyn std::error::Error + Send + Sync>> {
        let mut processed_files = Vec::new();
        let local_metadata = Self::load_local_metadata()?;

        for file in github_files {
            if Self::should_process_file(&file, &local_metadata) {
                debug!("Processing file: {}", file.name);
                let stripped_content = Self::strip_double_brackets(&file.content);
                let processed_content = Self::process_against_topics(&stripped_content, &settings.topics);
                processed_files.push(ProcessedFile { content: processed_content.clone() });

                // Update local metadata
                let new_metadata = Metadata {
                    file_name: file.name.clone(),
                    last_modified: chrono::Utc::now(),
                    processed_file: processed_content,
                    original_file: file.content,
                };
                Self::save_file_metadata(new_metadata)?;
            } else {
                debug!("Skipping file: {}", file.name);
            }
        }

        Ok(processed_files)
    }

    fn should_process_file(file: &GithubFile, local_metadata: &HashMap<String, Metadata>) -> bool {
        let first_line = file.content.lines().next().unwrap_or("").trim();
        if first_line != "public:: true" {
            return false;
        }

        let local_sha = local_metadata.get(&file.name).map(|m| Self::calculate_sha1(&m.original_file));
        let github_sha = Self::calculate_sha1(&file.content);

        local_sha.map_or(true, |local| local != github_sha)
    }

    fn calculate_sha1(content: &str) -> String {
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn load_local_metadata() -> Result<HashMap<String, Metadata>, Box<dyn std::error::Error + Send + Sync>> {
        // Implement loading of local metadata from a file or database
        // For now, we'll return an empty HashMap
        Ok(HashMap::new())
    }

    fn strip_double_brackets(content: &str) -> String {
        let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
        re.replace_all(content, "$1").to_string()
    }

    fn process_against_topics(content: &str, topics: &[String]) -> String {
        // Implement your logic to process the content against topics here
        // This is a placeholder implementation
        let mut processed_content = content.to_string();
        for topic in topics {
            if content.contains(topic) {
                processed_content.push_str(&format!("\nRelated to topic: {}", topic));
            }
        }
        processed_content
    }

    pub fn save_file_metadata(metadata: Metadata) -> Result<(), std::io::Error> {
        info!("Saving metadata for file: {}", metadata.file_name);
        // Implement metadata saving logic here
        Ok(())
    }
}

pub struct RealGitHubService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
}

impl RealGitHubService {
    pub fn new() -> Self {
        dotenv().ok();
        let token = env::var("GITHUB_ACCESS_TOKEN").expect("GITHUB_ACCESS_TOKEN must be set in .env");
        let owner = env::var("GITHUB_OWNER").expect("GITHUB_OWNER must be set in .env");
        let repo = env::var("GITHUB_REPO").expect("GITHUB_REPO must be set in .env");
        let base_path = env::var("GITHUB_DIRECTORY").expect("GITHUB_DIRECTORY must be set in .env");

        Self {
            client: Client::new(),
            token,
            owner,
            repo,
            base_path,
        }
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

        let contents: Vec<serde_json::Value> = serde_json::from_str(&response_body)?;
        Ok(contents)
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
}

#[async_trait]
impl GitHubService for RealGitHubService {
    async fn fetch_files(&self) -> Result<Vec<GithubFile>, Box<dyn std::error::Error + Send + Sync>> {
        let mut github_files = Vec::new();
        let mut directories_to_process = vec![self.base_path.clone()];

        while let Some(current_path) = directories_to_process.pop() {
            let contents = self.fetch_directory_contents(&current_path).await?;

            for item in contents {
                let name = item["name"].as_str().unwrap_or("");
                let item_type = item["type"].as_str().unwrap_or("");
                let path = item["path"].as_str().unwrap_or("");

                if item_type == "dir" {
                    directories_to_process.push(path.to_string());
                } else if item_type == "file" && name.ends_with(".md") {
                    if let Some(download_url) = item["download_url"].as_str() {
                        debug!("Fetching content for file: {}", name);
                        let content = self.fetch_file_content(download_url).await?;
                        let sha = item["sha"].as_str().unwrap_or("").to_string();

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
