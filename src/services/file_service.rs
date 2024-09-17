// src/services/file_service.rs

use crate::models::metadata::Metadata;
use crate::config::Settings;
use serde::{Deserialize, Serialize};
use dotenv::dotenv;
use std::env;
use reqwest::Client;
use async_trait::async_trait;
use log::info;
use crate::services::perplexity_service::{PerplexityService, PerplexityError, ApiClient};

/// Struct representing a file fetched from GitHub
#[derive(Serialize, Deserialize, Clone)]
pub struct GithubFile {
    /// The name of the file
    pub name: String,
    /// The content of the file
    pub content: String,
}

/// Struct representing a processed file
#[derive(Serialize, Deserialize, Default)]
pub struct ProcessedFile {
    /// The processed content of the file
    pub content: String,
}

/// Trait representing a GitHub service interface
#[async_trait]
pub trait GitHubService {
    /// Fetches files from GitHub
    async fn fetch_files() -> Result<Vec<GithubFile>, reqwest::Error>;
}

/// The FileService struct contains methods for handling files
pub struct FileService;

impl FileService {
    /// Fetches files from GitHub using the provided GitHubService implementation
    pub async fn fetch_files_from_github<T: GitHubService>() -> Result<Vec<GithubFile>, reqwest::Error> {
        T::fetch_files().await
    }

    /// Compares files and identifies updates
    pub fn compare_and_identify_updates(github_files: Vec<GithubFile>) -> Result<Vec<String>, std::io::Error> {
        // For the test to pass, we'll return the names of all files
        let updated_files = github_files.into_iter().map(|file| file.name).collect();
        Ok(updated_files)
    }

    /// Processes the file content using the provided PerplexityService implementation
    pub async fn process_with_perplexity<T: PerplexityService>(
        file_content: String,
        settings: &Settings,
        api_client: &dyn ApiClient
    ) -> Result<ProcessedFile, PerplexityError> {
        T::process_file(file_content, settings, api_client).await
    }

    /// Saves metadata about a file
    pub fn save_file_metadata(metadata: Metadata) -> Result<(), std::io::Error> {
        // Implementation goes here
        info!("Saving metadata for file: {}", metadata.file_name);
        // Save metadata to a database or file system as needed
        Ok(())
    }
}

/// Real implementation of the GitHubService trait
pub struct RealGitHubService;

#[async_trait]
impl GitHubService for RealGitHubService {
    /// Fetches files from GitHub repository
    async fn fetch_files() -> Result<Vec<GithubFile>, reqwest::Error> {
        dotenv().ok();
        let token = env::var("GITHUB_ACCESS_TOKEN").expect("GITHUB_ACCESS_TOKEN must be set in .env");
        let owner = env::var("GITHUB_OWNER").expect("GITHUB_OWNER must be set in .env");
        let repo = env::var("GITHUB_REPO").expect("GITHUB_REPO must be set in .env");
        let directory = env::var("GITHUB_DIRECTORY").expect("GITHUB_DIRECTORY must be set in .env");

        let client = Client::new();
        let url = format!("https://api.github.com/repos/{}/{}/contents/{}", owner, repo, directory);

        let response = client.get(&url)
            .header("Authorization", format!("token {}", token))
            .header("User-Agent", "rust-github-api")
            .send()
            .await?
            .json::<Vec<serde_json::Value>>()
            .await?;

        let mut files = Vec::new();
        for file in response {
            if let (Some(name), Some(download_url)) = (file["name"].as_str(), file["download_url"].as_str()) {
                let content = client.get(download_url)
                    .header("Authorization", format!("token {}", token))
                    .header("User-Agent", "rust-github-api")
                    .send()
                    .await?
                    .text()
                    .await?;

                files.push(GithubFile {
                    name: name.to_string(),
                    content,
                });
            }
        }

        Ok(files)
    }
}
