use super::api::GitHubClient;
use super::types::{CreateBranchRequest, CreatePullRequest, UpdateFileRequest, PullRequestResponse};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use log::{error, info};
use std::error::Error;
use chrono::Utc;

/// Handles GitHub Pull Request operations
use std::sync::Arc;

pub struct PullRequestAPI {
    client: Arc<GitHubClient>,
}

impl PullRequestAPI {
    /// Create a new PullRequestAPI instance
    pub fn new(client: Arc<GitHubClient>) -> Self {
        Self { client }
    }

    /// Create a pull request for a file update
    pub async fn create_pull_request(
        &self,
        file_name: &str,
        content: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let timestamp = Utc::now().timestamp();
        let branch_name = format!("update-{}-{}", file_name.replace(".md", ""), timestamp);
        
        let main_sha = self.get_main_branch_sha().await?;
        self.create_branch(&branch_name, &main_sha).await?;
        
        let file_path = format!("{}/{}", self.client.base_path(), file_name);
        let new_sha = self.update_file(&file_path, content, &branch_name, original_sha).await?;
        
        let url = format!(
            "https://api.github.com/repos/{}/{}/pulls",
            self.client.owner(), self.client.repo()
        );

        let pr_body = CreatePullRequest {
            title: format!("Update: {}", file_name),
            head: branch_name,
            base: "main".to_string(),
            body: format!(
                "This PR updates content for {}.\n\nOriginal SHA: {}\nNew SHA: {}",
                file_name, original_sha, new_sha
            ),
        };

        let response = self.client.client()
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .json(&pr_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to create PR: {}", error_text);
            return Err(format!("GitHub API error: {}", error_text).into());
        }

        let pr_response: PullRequestResponse = response.json().await?;
        info!("Created PR: {}", pr_response.html_url);
        Ok(pr_response.html_url)
    }

    /// Get the SHA of the main branch
    async fn get_main_branch_sha(&self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/git/ref/heads/main",
            self.client.owner(), self.client.repo()
        );

        let response = self.client.client()
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to get main branch SHA: {}", error_text);
            return Err(format!("GitHub API error: {}", error_text).into());
        }

        let response_json: serde_json::Value = response.json().await?;
        Ok(response_json["object"]["sha"]
            .as_str()
            .ok_or_else(|| "SHA not found in response".to_string())?
            .to_string())
    }

    /// Create a new branch
    async fn create_branch(&self, branch_name: &str, sha: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/git/refs",
            self.client.owner(), self.client.repo()
        );

        let body = CreateBranchRequest {
            ref_name: format!("refs/heads/{}", branch_name),
            sha: sha.to_string(),
        };

        let response = self.client.client()
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to create branch: {}", error_text);
            return Err(format!("GitHub API error: {}", error_text).into());
        }

        Ok(())
    }

    /// Update a file in a branch
    async fn update_file(
        &self,
        file_path: &str,
        content: &str,
        branch_name: &str,
        original_sha: &str,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            self.client.owner(), self.client.repo(), file_path
        );

        let encoded_content = BASE64.encode(content);
        
        let body = UpdateFileRequest {
            message: format!("Update {}", file_path),
            content: encoded_content,
            sha: original_sha.to_string(),
            branch: branch_name.to_string(),
        };

        let response = self.client.client()
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.client.token()))
            .header("Accept", "application/vnd.github+json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("Failed to update file: {}", error_text);
            return Err(format!("GitHub API error: {}", error_text).into());
        }

        let response_json: serde_json::Value = response.json().await?;
        Ok(response_json["content"]["sha"]
            .as_str()
            .ok_or_else(|| "SHA not found in response".to_string())?
            .to_string())
    }
}