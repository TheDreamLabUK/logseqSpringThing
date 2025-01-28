//! GitHub service module providing API interactions for content and pull requests
//! 
//! This module is split into:
//! - Content API: Handles fetching and checking markdown files
//! - Pull Request API: Manages creation and updates of pull requests
//! - Common types and error handling

mod api;
mod content;
mod pr;
mod types;

pub use api::GitHubClient;
pub use content::ContentAPI;
pub use pr::PullRequestAPI;
pub use types::{GitHubError, GitHubFile, GitHubFileMetadata};

// Re-export commonly used types for convenience
pub use types::{ContentResponse, PullRequestResponse};