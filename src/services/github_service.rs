use reqwest::Client;
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use log::{info, error};
use std::error::Error;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

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
            .build()?;

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
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to get main branch SHA: {}", error_text);
            return Err(format!("Failed to get main branch SHA: {}", error_text).into());
        }

        let response_json: serde_json::Value = response.json().await?;
        Ok(response_json["object"]["sha"]
            .as_str()
            .ok_or("SHA not found")?
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
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to create branch: {}", error_text);
            return Err(format!("Failed to create branch: {}", error_text).into());
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
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to update file: {}", error_text);
            return Err(format!("Failed to update file: {}", error_text).into());
        }

        let file_response: FileResponse = response.json().await?;
        Ok(file_response.sha)
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
        
        // Get main branch SHA
        let main_sha = self.get_main_branch_sha().await?;
        
        // Create new branch
        self.create_branch(&branch_name, &main_sha).await?;
        
        // Update file in new branch
        let file_path = format!("{}/{}", self.base_path, file_name);
        let new_sha = self.update_file(&file_path, content, &branch_name, original_sha).await?;
        
        // Create pull request
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
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to create PR: {}", error_text);
            return Err(format!("Failed to create PR: {}", error_text).into());
        }

        let pr_response: serde_json::Value = response.json().await?;
        let pr_url = pr_response["html_url"]
            .as_str()
            .ok_or("PR URL not found")?
            .to_string();

        info!("Created PR: {}", pr_url);
        Ok(pr_url)
    }
}
