use reqwest::Client;
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use log::{info, error};
use std::error::Error;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

#[derive(Debug)]
pub enum GitHubError {
    ApiError(String),
    NetworkError(reqwest::Error),
    SerializationError(serde_json::Error),
    ValidationError(String),
    Base64Error(base64::DecodeError),
}

impl std::fmt::Display for GitHubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitHubError::ApiError(msg) => write!(f, "GitHub API error: {}", msg),
            GitHubError::NetworkError(e) => write!(f, "Network error: {}", e),
            GitHubError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            GitHubError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            GitHubError::Base64Error(e) => write!(f, "Base64 encoding error: {}", e),
        }
    }
}

impl Error for GitHubError {}

impl From<reqwest::Error> for GitHubError {
    fn from(err: reqwest::Error) -> Self {
        GitHubError::NetworkError(err)
    }
}

impl From<serde_json::Error> for GitHubError {
    fn from(err: serde_json::Error) -> Self {
        GitHubError::SerializationError(err)
    }
}

impl From<base64::DecodeError> for GitHubError {
    fn from(err: base64::DecodeError) -> Self {
        GitHubError::Base64Error(err)
    }
}

#[derive(Debug, Serialize)]
struct CreateBranchRequest {
    pub ref_name: String,
    pub sha: String,
}

#[derive(Debug, Serialize)]
struct CreatePullRequest {
    pub title: String,
    pub head: String,
    pub base: String,
    pub body: String,
}

#[derive(Debug, Serialize)]
struct UpdateFileRequest {
    pub message: String,
    pub content: String,
    pub sha: String,
    pub branch: String,
}

#[derive(Debug, Deserialize)]
struct FileResponse {
    pub sha: String,
}

#[async_trait]
pub trait GitHubPRService: Send + Sync {
    async fn create_pull_request(
        &self,
        file_name: &str,
        content: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>>;
}

pub struct RealGitHubPRService {
    client: Client,
    token: String,
    owner: String,
    repo: String,
    base_path: String,
}

impl RealGitHubPRService {
    pub fn new(
        token: String,
        owner: String,
        repo: String,
        base_path: String,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let client = Client::builder()
            .user_agent("rust-github-api")
            .build()
            .map_err(GitHubError::from)?;

        Ok(Self {
            client,
            token,
            owner,
            repo,
            base_path,
        })
    }

    async fn get_main_branch_sha(&self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/git/ref/heads/main",
            self.owner, self.repo
        );

        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(GitHubError::from)?;

        if !response.status().is_success() {
            let error_text = response.text().await.map_err(GitHubError::from)?;
            error!("Failed to get main branch SHA: {}", error_text);
            return Err(GitHubError::ApiError(error_text).into());
        }

        let response_json: serde_json::Value = response.json().await.map_err(GitHubError::from)?;
        Ok(response_json["object"]["sha"]
            .as_str()
            .ok_or_else(|| GitHubError::ValidationError("SHA not found".to_string()))?
            .to_string())
    }

    async fn create_branch(&self, branch_name: &str, sha: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/git/refs",
            self.owner, self.repo
        );

        let body = CreateBranchRequest {
            ref_name: format!("refs/heads/{}", branch_name),
            sha: sha.to_string(),
        };

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .json(&body)
            .send()
            .await
            .map_err(GitHubError::from)?;

        if !response.status().is_success() {
            let error_text = response.text().await.map_err(GitHubError::from)?;
            error!("Failed to create branch: {}", error_text);
            return Err(GitHubError::ApiError(error_text).into());
        }

        Ok(())
    }

    async fn update_file(
        &self,
        file_path: &str,
        content: &str,
        branch_name: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner, self.repo, file_path
        );

        let encoded_content = BASE64.encode(content);
        
        let body = UpdateFileRequest {
            message: format!("Update {} with Perplexity-enhanced content", file_path),
            content: encoded_content,
            sha: original_sha.to_string(),
            branch: branch_name.to_string(),
        };

        let response = self.client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .json(&body)
            .send()
            .await
            .map_err(GitHubError::from)?;

        if !response.status().is_success() {
            let error_text = response.text().await.map_err(GitHubError::from)?;
            error!("Failed to update file: {}", error_text);
            return Err(GitHubError::ApiError(error_text).into());
        }

        let file_response: FileResponse = response.json().await.map_err(GitHubError::from)?;
        Ok(file_response.sha)
    }

    #[allow(dead_code)]
    async fn get_contents_url(&self, path: &str) -> String {
        let full_path = if path.is_empty() {
            self.base_path.clone()
        } else {
            format!("{}/{}", self.base_path.trim_matches('/'), path.trim_matches('/'))
        };

        format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.owner,
            self.repo,
            full_path
        )
    }

    fn get_api_path(&self) -> String {
        self.base_path.trim_matches('/').to_string()
    }

    fn get_full_path(&self, path: &str) -> String {
        let base = self.base_path.trim_matches('/');
        let path = path.trim_matches('/');
        
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
}

#[async_trait]
impl GitHubPRService for RealGitHubPRService {
    async fn create_pull_request(
        &self,
        file_name: &str,
        content: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let timestamp = chrono::Utc::now().timestamp();
        let branch_name = format!("perplexity-update-{}-{}", file_name.replace(".md", ""), timestamp);
        
        let main_sha = self.get_main_branch_sha().await?;
        self.create_branch(&branch_name, &main_sha).await?;
        
        let file_path = format!("{}/{}", self.base_path, file_name);
        let new_sha = self.update_file(&file_path, content, &branch_name, original_sha).await?;
        
        let url = format!(
            "https://api.github.com/repos/{}/{}/pulls",
            self.owner, self.repo
        );

        let pr_body = CreatePullRequest {
            title: format!("Perplexity Enhancement: {}", file_name),
            head: branch_name,
            base: "main".to_string(),
            body: format!(
                "This PR contains Perplexity-enhanced content for {}.\n\nOriginal SHA: {}\nNew SHA: {}",
                file_name, original_sha, new_sha
            ),
        };

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .json(&pr_body)
            .send()
            .await
            .map_err(GitHubError::from)?;

        if !response.status().is_success() {
            let error_text = response.text().await.map_err(GitHubError::from)?;
            error!("Failed to create PR: {}", error_text);
            return Err(GitHubError::ApiError(error_text).into());
        }

        let pr_response: serde_json::Value = response.json().await.map_err(GitHubError::from)?;
        let pr_url = pr_response["html_url"]
            .as_str()
            .ok_or_else(|| GitHubError::ValidationError("PR URL not found".to_string()))?
            .to_string();

        info!("Created PR: {}", pr_url);
        Ok(pr_url)
    }
}
